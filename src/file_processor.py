"""
文件上传识别处理（本地 FunASR 一体化管线）：
- paraformer-zh：语音识别（支持热词）
- fsmn-vad：语音活动检测
- ct-punc：标点恢复
- cam++：说话人分离

实时录音走远程 API，文件上传走本地模型（更准确）
"""
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config
from transcript_buffer import ASRResult

logger = logging.getLogger(__name__)

# 模型根目录
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "iic"
if not _MODELS_DIR.is_dir():
    _MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "iic"

# 本地 FunASR 模型单例
_local_asr_model = None


def _get_local_asr_model():
    """延迟加载本地 FunASR 一体化模型（paraformer + vad + punc + spk）"""
    global _local_asr_model
    if _local_asr_model is not None:
        return _local_asr_model

    os.environ["MODELSCOPE_OFFLINE"] = "1"
    from funasr import AutoModel

    # 模型路径
    asr_path = _MODELS_DIR / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_path = _MODELS_DIR / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_path = _MODELS_DIR / "punc_ct-transformer_cn-en-common-vocab471067-large"
    spk_path = _MODELS_DIR / "speech_campplus_sv_zh-cn_16k-common"

    # 检查模型是否存在
    missing = []
    if not asr_path.is_dir():
        missing.append(f"paraformer-zh: {asr_path}")
    if not vad_path.is_dir():
        missing.append(f"fsmn-vad: {vad_path}")
    if not punc_path.is_dir():
        missing.append(f"ct-punc: {punc_path}")
    if not spk_path.is_dir():
        missing.append(f"cam++: {spk_path}")

    if missing:
        logger.error(f"本地模型缺失: {missing}")
        return None

    try:
        _local_asr_model = AutoModel(
            model=str(asr_path),
            vad_model=str(vad_path),
            punc_model=str(punc_path),
            spk_model=str(spk_path),
            vad_kwargs={"max_single_segment_time": config.ASR_VAD_MAX_SEGMENT_MS},
            disable_update=True,
        )
        logger.info("本地 FunASR 一体化模型加载成功 (paraformer + vad + punc + cam++)")
        return _local_asr_model
    except Exception as e:
        logger.error(f"本地 FunASR 模型加载失败: {e}")
        return None


def transcribe_file_with_diarization(
    audio_path: str,
    client=None,  # 保留接口兼容，但文件上传不再用远程 client
    n_speakers: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[list[ASRResult], dict]:
    """
    使用本地 FunASR 一体化管线处理上传文件：
    paraformer-zh + fsmn-vad + ct-punc + cam++

    支持热词、说话人分离、标点恢复
    """
    if progress_callback:
        progress_callback(0, 1, "正在加载本地模型...")

    model = _get_local_asr_model()
    if model is None:
        raise RuntimeError("本地 FunASR 模型加载失败，请检查 models/ 目录")

    if progress_callback:
        progress_callback(0, 1, "模型已就绪，开始识别...")

    # 构建热词参数
    hotword = config.ASR_CONTEXT if hasattr(config, "ASR_CONTEXT") and config.ASR_CONTEXT else ""

    try:
        # 先用 librosa 加载音频为 numpy 数组（支持多种格式，不依赖 ffmpeg）
        import librosa
        audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio_data = audio_data.astype(np.float32)

        # FunASR 一体化调用：传入 numpy 数组避免 torchaudio/ffmpeg 依赖
        generate_kwargs = {
            "input": audio_data,
            "batch_size_s": 300,
        }
        if hotword:
            generate_kwargs["hotword"] = hotword

        res = model.generate(**generate_kwargs)

        if not res or not isinstance(res, list) or len(res) == 0:
            return [], {"total_duration": 0, "segments": 0, "speakers": 0}

        # 解析结果
        results = []
        speaker_set = set()

        for item in res:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "").strip()
            if not text:
                continue

            # 解析时间戳
            timestamp = item.get("timestamp", [[0, 0]])
            if timestamp and len(timestamp) > 0:
                start_ms = timestamp[0][0] if isinstance(timestamp[0], (list, tuple)) else 0
            else:
                start_ms = 0

            # 解析说话人
            sentence_info = item.get("sentence_info", [])
            if sentence_info:
                # sentence_info 包含每句的详细信息（文本、时间戳、说话人）
                for sent in sentence_info:
                    sent_text = sent.get("text", "").strip()
                    if not sent_text:
                        continue
                    spk = sent.get("spk", -1)
                    sent_start = sent.get("start", 0)
                    sent_end = sent.get("end", 0)
                    speaker_set.add(spk)
                    results.append(ASRResult(
                        text=sent_text,
                        timestamp=sent_start / 1000.0,
                        duration=(sent_end - sent_start) / 1000.0,
                        speaker=spk,
                        is_partial=False,
                    ))
            else:
                # 没有 sentence_info，用整体结果
                results.append(ASRResult(
                    text=text,
                    timestamp=start_ms / 1000.0,
                    duration=0,
                    speaker=-1,
                    is_partial=False,
                ))

        # 计算统计
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(audio) / 16000.0

        stats = {
            "total_duration": total_duration,
            "segments": len(results),
            "recognized": len(results),
            "failed": 0,
            "speakers": len(speaker_set) if speaker_set != {-1} else 1,
        }

        if progress_callback:
            progress_callback(1, 1, f"识别完成：{len(results)} 句，{stats['speakers']} 位说话人")

        return results, stats

    except Exception as e:
        logger.exception("本地 FunASR 识别失败")
        raise RuntimeError(f"识别失败: {e}")


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
