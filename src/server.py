"""
FastAPI 后端服务

- GET  /              -> 返回 index.html
- POST /api/transcribe-chunk   -> 实时识别（前端 MediaRecorder 录的音频片段）
- POST /api/transcribe-file    -> 文件上传：说话人分离 + 识别
- POST /api/summary            -> 根据转写文本生成会议纪要（流式 SSE）
"""
import io
import json
import logging
import sys
import tempfile
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

app = FastAPI(title="远程 Qwen3-ASR 会议记录")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ─────────────────────────────────────────
# 实时识别（前端按时间窗口切片发上来）
# ─────────────────────────────────────────

@app.post("/api/transcribe-chunk")
async def transcribe_chunk(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    timestamp: float = Form(0.0),
):
    """
    前端 MediaRecorder 切片上传。
    任意常见音频格式（webm/ogg/mp4/wav 等），直接转发到远程 ASR。
    """
    data = await audio.read()
    if not data:
        return {"text": "", "timestamp": timestamp}

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
            tmp_path, language=language, context=config.ASR_CONTEXT or None
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
):
    """整文件识别 + 说话人分离，进度通过 SSE 流推送。"""
    data = await audio.read()
    if not data:
        raise HTTPException(400, "空文件")

    suffix = _suffix_for_mime(audio.content_type or "", audio.filename)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    def event_stream():
        try:
            progress_state = {"done": 0, "total": 0, "msg": ""}

            def progress_cb(done: int, total: int, msg: str):
                progress_state["done"] = done
                progress_state["total"] = total
                progress_state["msg"] = msg

            # 先发一个开始事件
            yield _sse({"event": "start", "filename": audio.filename})

            # 由于 progress 是同步回调，这里改为一边跑一边发：
            # 为简化，先全部跑完再一次性返回（前端用单个 fetch + json 也可，但流式更好看）。
            results, stats = transcribe_file_with_diarization(
                tmp_path,
                _asr_client,
                n_speakers=int(n_speakers),
                progress_callback=progress_cb,
            )
            transcript_text = format_transcript(results)
            llm_input = format_transcript_for_llm(results)
            yield _sse({
                "event": "transcript",
                "text": transcript_text,
                "llm_input": llm_input,
                "stats": stats,
            })
            yield _sse({"event": "done"})
        except Exception as e:
            logger.exception("文件识别失败")
            yield _sse({"event": "error", "message": str(e)})
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
