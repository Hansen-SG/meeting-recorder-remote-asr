"""
FastAPI 后端服务

- GET  /              -> 返回 index.html
- POST /api/transcribe-chunk   -> 实时识别（前端 MediaRecorder 录的音频片段）
- POST /api/transcribe-file    -> 文件上传：说话人分离 + 识别（本地 FunASR 并行）
- POST /api/summary            -> 根据转写文本生成会议纪要（流式 SSE）
"""
import io
import json
import logging
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from file_processor import (
    transcribe_file_with_diarization,
    format_transcript,
    format_transcript_for_llm,
)
from llm_summary import LLMSummaryGenerator
from qwen_asr import Qwen3ASRClient, Qwen3ASRError

logger = logging.getLogger(__name__)

# 静态文件根目录
ROOT_DIR = SRC_DIR.parent
STATIC_DIR = ROOT_DIR / "web"
INDEX_HTML = STATIC_DIR / "index.html"

# ─────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────
_asr_client = Qwen3ASRClient()
_summary_gen = LLMSummaryGenerator()

# ─────────────────────────────────────────
# 任务队列状态（用于前端排队提示）
# ─────────────────────────────────────────
_queue_lock = threading.Lock()
_queue_processing = 0  # 当前正在处理的文件数
_queue_waiting = 0     # 排队等待的文件数

app = FastAPI(title="远程 Qwen3-ASR 会议记录")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_preload_model():
    """服务启动时预加载模型：主进程模型 + 6进程预热池"""
    from file_processor import _get_local_model, init_worker_pool
    try:
        logger.info("正在预加载本地 FunASR 模型（主进程）...")
        _get_local_model()
        logger.info("主进程模型预加载完成")
    except Exception as e:
        logger.warning(f"主进程模型预加载失败: {e}")

    try:
        init_worker_pool()
    except Exception as e:
        logger.warning(f"进程池预热失败（大文件将降级为顺序处理）: {e}")


# ─────────────────────────────────────────
# 静态页面
# ─────────────────────────────────────────

@app.get("/")
async def index():
    if not INDEX_HTML.exists():
        return JSONResponse(
            {"error": f"前端文件不存在: {INDEX_HTML}"}, status_code=500
        )
    return FileResponse(str(INDEX_HTML))


@app.get("/static/{filename}")
async def serve_static(filename: str):
    """提供静态资源（css/js）"""
    target = STATIC_DIR / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Not Found")
    return FileResponse(str(target))


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "asr_model": config.ASR_MODEL,
        "asr_base": config.ASR_API_BASE,
        "llm_model": config.LLM_MODEL,
        "llm_base": config.LLM_API_BASE,
    }


@app.get("/api/queue-status")
async def queue_status():
    """返回当前文件处理队列状态"""
    with _queue_lock:
        processing = _queue_processing
        waiting = _queue_waiting
    return {
        "processing": processing,
        "waiting": waiting,
        "total": processing + waiting,
    }


# ─────────────────────────────────────────
# 实时识别（前端按时间窗口切片发上来）
# ─────────────────────────────────────────

@app.get("/api/hotwords")
async def get_hotwords():
    """获取用户可配置的热词（不包含内置热词）"""
    return {"hotwords": config.ASR_CONTEXT or ""}


@app.post("/api/transcribe-chunk")
async def transcribe_chunk(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    timestamp: float = Form(0.0),
    hotwords: str = Form(""),
):
    """
    前端 MediaRecorder 切片上传。
    任意常见音频格式（webm/ogg/mp4/wav 等），直接转发到远程 ASR。
    hotwords: 空格或逗号分隔的热词列表，为空则使用 config 默认值。
    """
    data = await audio.read()
    if not data:
        return {"text": "", "timestamp": timestamp}

    # 合并内置热词 + 用户热词
    context = config.get_hotwords(hotwords.strip())

    # 检测 MIME 类型
    mime = audio.content_type or "audio/webm"

    # 保存到临时文件让 Qwen ASR 处理（也可直接 bytes，但保留扩展名更稳）
    suffix = _suffix_for_mime(mime, audio.filename)
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = _asr_client.transcribe_file(
            tmp_path, language=language, context=context
        )
        text = (result.get("text") or "").strip()
        return {"text": text, "timestamp": timestamp}
    except Qwen3ASRError as e:
        logger.warning(f"实时识别失败: {e}")
        return JSONResponse({"error": str(e), "timestamp": timestamp}, status_code=502)
    except Exception as e:
        logger.exception("实时识别异常")
        return JSONResponse({"error": str(e), "timestamp": timestamp}, status_code=500)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────
# 文件上传：说话人分离 + 识别
# ─────────────────────────────────────────

@app.post("/api/transcribe-file")
async def transcribe_file(
    audio: UploadFile = File(...),
    n_speakers: int = Form(0),
    hotwords: str = Form(""),
):
    """整文件识别 + 说话人分离，进度通过 SSE 流推送。使用本地 FunASR 并行处理。"""
    data = await audio.read()
    if not data:
        raise HTTPException(400, "空文件")

    # 队列计数：进入排队
    global _queue_processing, _queue_waiting
    with _queue_lock:
        _queue_waiting += 1

    suffix = _suffix_for_mime(audio.content_type or "", audio.filename)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    def event_stream():
        global _queue_processing, _queue_waiting
        progress_queue = queue.Queue()

        # 队列计数：从排队转为处理中
        with _queue_lock:
            _queue_waiting = max(0, _queue_waiting - 1)
            _queue_processing += 1

        def progress_cb(done: int, total: int, msg: str):
            progress_queue.put({"event": "progress", "done": done, "total": total, "msg": msg})

        def run_processing():
            try:
                # 合并热词：前端传入优先，为空时用配置默认值
                hw = hotwords.strip() if hotwords.strip() else None
                results, stats = transcribe_file_with_diarization(
                    tmp_path,
                    client=None,
                    n_speakers=int(n_speakers),
                    progress_callback=progress_cb,
                    hotwords_override=hw,
                )
                transcript_text = format_transcript(results)
                llm_input = format_transcript_for_llm(results)
                progress_queue.put({
                    "event": "_result",
                    "text": transcript_text,
                    "llm_input": llm_input,
                    "stats": stats,
                })
            except Exception as e:
                logger.exception("文件识别失败")
                progress_queue.put({"event": "_error", "message": str(e)})
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        thread = threading.Thread(target=run_processing, daemon=True)
        thread.start()

        yield _sse({"event": "start", "filename": audio.filename})

        while True:
            try:
                msg = progress_queue.get(timeout=600)
            except queue.Empty:
                yield _sse({"event": "error", "message": "处理超时"})
                break

            if msg["event"] == "progress":
                yield _sse(msg)
            elif msg["event"] == "_result":
                yield _sse({
                    "event": "transcript",
                    "text": msg["text"],
                    "llm_input": msg["llm_input"],
                    "stats": msg["stats"],
                })
                break
            elif msg["event"] == "_error":
                yield _sse({"event": "error", "message": msg["message"]})
                break

        # 队列计数：处理完成
        with _queue_lock:
            _queue_processing = max(0, _queue_processing - 1)

        yield _sse({"event": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ─────────────────────────────────────────
# 会议纪要生成（流式）
# ─────────────────────────────────────────

@app.post("/api/summary")
async def summary(transcript: str = Form(...), title: Optional[str] = Form(None)):
    """流式 SSE 返回纪要 token"""
    if not transcript or len(transcript.strip()) < 20:
        return JSONResponse(
            {"error": "转写内容太短"}, status_code=400
        )

    def event_stream():
        try:
            yield _sse({"event": "start"})
            for token in _summary_gen.generate_streaming(transcript, title):
                yield _sse({"event": "token", "text": token})
            yield _sse({"event": "done"})
        except Exception as e:
            logger.exception("纪要生成失败")
            yield _sse({"event": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────
# 导出云文档
# ─────────────────────────────────────────

@app.post("/api/export")
async def export_to_doc(
    transcript: str = Form(""),
    summary: str = Form(""),
    title: Optional[str] = Form(None),
):
    """将转写结果和会议纪要导出为飞书云文档"""
    if not transcript and not summary:
        return JSONResponse({"error": "无内容可导出"}, status_code=400)

    try:
        from feishu_export import FeishuExporter, FeishuExportError

        exporter = FeishuExporter()

        # 组合导出内容
        content_parts = []
        if summary:
            content_parts.append(summary)
        if transcript:
            content_parts.append("\n---\n\n## 完整转写记录\n\n")
            content_parts.append(transcript)

        full_content = "\n".join(content_parts)
        doc_title = title or "会议记录"

        result = exporter.export_summary(doc_title, full_content)
        logger.info(f"导出飞书文档成功: {result.get('url', '')}")
        return {"ok": True, "url": result["url"], "document_id": result["document_id"]}

    except Exception as e:
        logger.exception("导出飞书文档失败")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _suffix_for_mime(mime: str, filename: Optional[str]) -> str:
    """根据 MIME 或文件名得到合适的后缀"""
    if filename:
        suf = Path(filename).suffix.lower()
        if suf in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".mp4"):
            return suf
    m = (mime or "").lower()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/aac": ".aac",
    }
    return mapping.get(m, ".webm")
