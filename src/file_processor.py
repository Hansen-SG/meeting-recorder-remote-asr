"""
文件上传识别处理（本地 FunASR 一体化管线 + 预热进程池并行）：
- paraformer-zh：语音识别（支持热词）
- fsmn-vad：语音活动检测
- ct-punc：标点恢复
- cam++：说话人分离

架构：
- 服务启动时预创建 6 个 worker 进程，每个加载独立模型实例
- 大文件（>10分钟）拆分为 10 分钟块，分发到进程池并行处理
- 短文件用主进程全局模型处理（无进程间通信开销）

实时录音走远程 Qwen3-ASR-Flash API（不经过本模块）。
"""
import logging
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config
from transcript_buffer import ASRResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 全局模型单例（主进程，用于短文件）
# ─────────────────────────────────────────
_local_model = None


def _get_local_model():
    """延迟加载本地 FunASR 模型，主进程全局单例"""
    global _local_model
    if _local_model is not None:
        return _local_model

    os.environ["MODELSCOPE_OFFLINE"] = "1"

    try:
        from funasr import AutoModel
    except ImportError as e:
        raise RuntimeError(f"funasr 未安装: {e}")

    model_dir = Path(config.MODELS_DIR)
    asr_path = model_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_path = model_dir / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_path = model_dir / "punc_ct-transformer_cn-en-common-vocab471067-large"
    spk_path = model_dir / "speech_campplus_sv_zh-cn_16k-common"

    for p, name in [(asr_path, "paraformer"), (vad_path, "vad"), (punc_path, "punc"), (spk_path, "spk")]:
        if not p.is_dir():
            raise RuntimeError(f"模型缺失 ({name}): {p}")

    _local_model = AutoModel(
        model=str(asr_path),
        vad_model=str(vad_path),
        punc_model=str(punc_path),
        spk_model=str(spk_path),
        vad_kwargs={"max_single_segment_time": config.ASR_VAD_MAX_SEGMENT_MS},
        disable_update=True,
        disable_log=True,
    )

    seaco_weight = config.SEACO_WEIGHT
    if seaco_weight != 1.0:
        try:
            asr_model = _local_model.model
            original_decode = asr_model._seaco_decode_with_ASF

            def patched_decode(*args, **kwargs):
                kwargs.setdefault("seaco_weight", seaco_weight)
                return original_decode(*args, **kwargs)

            asr_model._seaco_decode_with_ASF = patched_decode
            logger.info(f"SeACo 热词权重设置为: {seaco_weight}")
        except Exception as e:
            logger.warning(f"设置 seaco_weight 失败（使用默认1.0）: {e}")

    logger.info("本地 FunASR 模型加载成功 (paraformer + vad + punc + cam++)")
    return _local_model


# ─────────────────────────────────────────
# 预热进程池（6 workers，启动时各自加载模型）
# ─────────────────────────────────────────
_worker_pool: Optional[ProcessPoolExecutor] = None


def _worker_init():
    """Worker 进程初始化：加载模型到进程本地全局变量"""
    global _worker_model
    os.environ["MODELSCOPE_OFFLINE"] = "1"

    from funasr import AutoModel

    model_dir = Path(os.environ.get("MODELS_DIR", config.MODELS_DIR))
    asr_path = model_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_path = model_dir / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_path = model_dir / "punc_ct-transformer_cn-en-common-vocab471067-large"
    spk_path = model_dir / "speech_campplus_sv_zh-cn_16k-common"

    _worker_model = AutoModel(
        model=str(asr_path),
        vad_model=str(vad_path),
        punc_model=str(punc_path),
        spk_model=str(spk_path),
        vad_kwargs={"max_single_segment_time": int(os.environ.get("ASR_VAD_MAX_SEGMENT_MS", "60000"))},
        disable_update=True,
        disable_log=True,
    )


def _worker_process_chunk(chunk_data: np.ndarray, hotwords: str) -> list[dict]:
    """Worker 进程中执行：用已加载的模型处理音频块"""
    global _worker_model

    generate_kwargs = {"input": chunk_data, "batch_size_s": 300}
    if hotwords:
        generate_kwargs["hotword"] = hotwords

    try:
        res = _worker_model.generate(**generate_kwargs)
    except Exception as e:
        return [{"error": f"识别失败: {e}"}]

    if not res or not isinstance(res, list) or len(res) == 0:
        return []

    results = []
    for item in res:
        if not isinstance(item, dict):
            continue
        sentence_info = item.get("sentence_info", [])
        if sentence_info:
            for sent in sentence_info:
                text = sent.get("text", "").strip()
                if text:
                    results.append({
                        "text": text,
                        "spk": sent.get("spk", -1),
                        "start": sent.get("start", 0),
                        "end": sent.get("end", 0),
                    })
        else:
            text = item.get("text", "").strip()
            if text:
                timestamp = item.get("timestamp", [[0, 0]])
                start_ms = timestamp[0][0] if timestamp and isinstance(timestamp[0], (list, tuple)) else 0
                results.append({
                    "text": text,
                    "spk": -1,
                    "start": start_ms,
                    "end": start_ms,
                })
    return results


def _worker_warmup():
    """Worker 预热任务（顶层函数，可 pickle）"""
    return "ready"


def init_worker_pool():
    """启动时调用：创建预热进程池（每个 worker 加载自己的模型）"""
    global _worker_pool
    n_workers = config.PARALLEL_WORKERS
    logger.info(f"正在创建 {n_workers} 进程预热池（每个进程加载独立模型）...")

    # 设置环境变量供 worker 读取
    os.environ["MODELS_DIR"] = config.MODELS_DIR
    os.environ["ASR_VAD_MAX_SEGMENT_MS"] = str(config.ASR_VAD_MAX_SEGMENT_MS)

    _worker_pool = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        mp_context=mp.get_context("spawn"),
    )

    # 预热：提交任务确保所有 worker 都初始化完成
    futures = [_worker_pool.submit(_worker_warmup) for _ in range(n_workers)]
    for f in futures:
        f.result(timeout=300)

    logger.info(f"预热进程池就绪：{n_workers} 个 worker 模型已加载")


# ─────────────────────────────────────────
# 音频处理辅助
# ─────────────────────────────────────────

def _run_generate(model, audio_data: np.ndarray, hotwords: str) -> list[dict]:
    """主进程：调用 model.generate 并解析 sentence_info"""
    generate_kwargs = {"input": audio_data, "batch_size_s": 300}
    if hotwords:
        generate_kwargs["hotword"] = hotwords

    res = model.generate(**generate_kwargs)

    if not res or not isinstance(res, list) or len(res) == 0:
        return []

    results = []
    for item in res:
        if not isinstance(item, dict):
            continue
        sentence_info = item.get("sentence_info", [])
        if sentence_info:
            for sent in sentence_info:
                text = sent.get("text", "").strip()
                if text:
                    results.append({
                        "text": text,
                        "spk": sent.get("spk", -1),
                        "start": sent.get("start", 0),
                        "end": sent.get("end", 0),
                    })
        else:
            text = item.get("text", "").strip()
            if text:
                timestamp = item.get("timestamp", [[0, 0]])
                start_ms = timestamp[0][0] if timestamp and isinstance(timestamp[0], (list, tuple)) else 0
                results.append({
                    "text": text,
                    "spk": -1,
                    "start": start_ms,
                    "end": start_ms,
                })
    return results


# ─────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────

def transcribe_file_with_diarization(
    audio_path: str,
    client=None,
    n_speakers: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    hotwords_override: Optional[str] = None,
) -> tuple[list[ASRResult], dict]:
    """
    使用本地 FunASR 一体化管线处理上传文件。
    短文件用主进程模型，长文件用预热进程池并行处理。
    """
    import librosa
    import subprocess
    import tempfile

    if progress_callback:
        progress_callback(0, 1, "正在加载音频文件...")

    # 1. 加载音频
    try:
        audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except ImportError:
            import shutil
            ffmpeg = shutil.which("ffmpeg")

        if not ffmpeg:
            raise RuntimeError("无法加载音频文件，且 ffmpeg 不可用。请上传 wav/mp3/flac 格式。")

        wav_tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_tmp],
                capture_output=True, check=True, timeout=120,
            )
            audio_data, _ = librosa.load(wav_tmp, sr=16000, mono=True)
        finally:
            try:
                os.unlink(wav_tmp)
            except Exception:
                pass

    audio_data = audio_data.astype(np.float32)
    total_duration = len(audio_data) / 16000.0

    if progress_callback:
        progress_callback(0, 1, f"音频加载完成 ({total_duration:.1f}s)，准备识别...")

    # 2. 分块
    chunk_seconds = config.FILE_CHUNK_SECONDS

    if total_duration <= chunk_seconds:
        chunks = [audio_data]
    else:
        chunk_samples = int(chunk_seconds * 16000)
        chunks = []
        offset = 0
        while offset < len(audio_data):
            end = min(offset + chunk_samples, len(audio_data))
            chunks.append(audio_data[offset:end])
            offset = end

    total_chunks = len(chunks)
    hotwords = config.get_hotwords(hotwords_override or "")

    logger.info(f"文件时长 {total_duration:.1f}s，拆分为 {total_chunks} 块")

    if progress_callback:
        progress_callback(0, total_chunks, f"开始识别（{total_chunks} 块）...")

    # 3. 处理
    chunk_results: dict[int, list[dict]] = {}
    errors = []

    if total_chunks == 1:
        # 短文件：主进程模型直接处理
        model = _get_local_model()
        result = _run_generate(model, chunks[0], hotwords)
        chunk_results[0] = result
        if progress_callback:
            progress_callback(1, 1, "识别完成")
    elif _worker_pool is not None:
        # 长文件：使用预热进程池并行处理
        logger.info(f"使用 {config.PARALLEL_WORKERS} 进程并行处理")
        try:
            future_to_idx = {}
            for i, chunk in enumerate(chunks):
                future = _worker_pool.submit(_worker_process_chunk, chunk, hotwords)
                future_to_idx[future] = i

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=600)
                    if result and isinstance(result[0], dict) and "error" in result[0]:
                        errors.append(f"块{idx}: {result[0]['error']}")
                        chunk_results[idx] = []
                    else:
                        chunk_results[idx] = result
                except Exception as e:
                    errors.append(f"块{idx}: {e}")
                    chunk_results[idx] = []

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_chunks, f"已完成 {completed}/{total_chunks} 块")
        except Exception as e:
            # 进程池损坏，降级为主进程顺序处理
            logger.warning(f"进程池异常（{e}），降级为顺序处理")
            model = _get_local_model()
            for i in range(total_chunks):
                if i in chunk_results:
                    continue
                if progress_callback:
                    progress_callback(i, total_chunks, f"正在识别第 {i+1}/{total_chunks} 块...")
                result = _run_generate(model, chunks[i], hotwords)
                chunk_results[i] = result
                if progress_callback:
                    progress_callback(i + 1, total_chunks, f"已完成 {i+1}/{total_chunks} 块")
    else:
        # 进程池未就绪，降级为主进程顺序处理
        logger.warning("进程池未就绪，降级为顺序处理")
        model = _get_local_model()
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, total_chunks, f"正在识别第 {i+1}/{total_chunks} 块...")
            result = _run_generate(model, chunk, hotwords)
            chunk_results[i] = result
            if progress_callback:
                progress_callback(i + 1, total_chunks, f"已完成 {i+1}/{total_chunks} 块")

    # 4. 合并结果
    all_results: list[ASRResult] = []
    speaker_set = set()

    for chunk_idx in range(total_chunks):
        items = chunk_results.get(chunk_idx, [])
        time_offset_ms = int(chunk_idx * chunk_seconds * 1000)

        for item in items:
            spk = item.get("spk", -1)
            start_ms = item["start"] + time_offset_ms
            end_ms = item["end"] + time_offset_ms

            all_results.append(ASRResult(
                text=item["text"],
                timestamp=start_ms / 1000.0,
                duration=(end_ms - start_ms) / 1000.0,
                speaker=spk,
                is_partial=False,
            ))
            if spk >= 0:
                speaker_set.add(spk)

    stats = {
        "total_duration": total_duration,
        "segments": len(all_results),
        "recognized": len(all_results),
        "failed": len(errors),
        "speakers": len(speaker_set - {-1}) if -1 in speaker_set else len(speaker_set),
        "chunks": total_chunks,
    }

    if errors:
        logger.warning(f"部分块识别失败: {errors}")

    if progress_callback:
        progress_callback(
            total_chunks, total_chunks,
            f"识别完成：{len(all_results)} 句，{stats['speakers']} 位说话人"
        )

    return all_results, stats


# ─────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────

def format_transcript(results: list[ASRResult]) -> str:
    """格式化为可读文本（带时间戳和说话人）"""
    lines = []
    for r in results:
        ts_min = int(r.timestamp) // 60
        ts_sec = int(r.timestamp) % 60
        ts_str = f"[{ts_min:02d}:{ts_sec:02d}]"
        spk_str = f" 说话人{r.speaker}" if r.speaker >= 0 else ""
        lines.append(f"{ts_str}{spk_str} {r.text}")
    return "\n".join(lines)


def format_transcript_for_llm(results: list[ASRResult]) -> str:
    """格式化为 LLM 输入格式（突出说话人）"""
    lines = []
    for r in results:
        if r.speaker >= 0:
            lines.append(f"[说话人{r.speaker}] {r.text}")
        else:
            lines.append(r.text)
    return "\n".join(lines)
