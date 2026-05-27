"""
说话人分离（fsmn-vad 神经网络分段 + cam++ 说话人嵌入 + 聚类）

流程：
1. fsmn-vad 神经网络精确切分语音段（比能量VAD精确得多）
2. cam++ 提取每段 192 维说话人嵌入
3. Agglomerative Clustering 聚类（自动选择簇数）
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 模型根目录
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "iic"
if not _MODELS_DIR.is_dir():
    _MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "iic"


@dataclass
class Segment:
    """语音段"""
    start: float       # 起始时间（秒）
    end: float         # 结束时间（秒）
    audio: np.ndarray  # float32 单声道音频
    speaker: int = -1  # 聚类后的说话人编号


# ─────────────────────────────────────────
# 模型单例
# ─────────────────────────────────────────
_campplus_model = None
_vad_model = None


def _get_vad_model():
    """延迟加载 fsmn-vad 模型"""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    os.environ["MODELSCOPE_OFFLINE"] = "1"
    from funasr import AutoModel

    vad_path = _MODELS_DIR / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    if not vad_path.is_dir():
        logger.warning(f"fsmn-vad 模型未找到: {vad_path}")
        return None

    try:
        _vad_model = AutoModel(model=str(vad_path), disable_update=True)
        logger.info(f"fsmn-vad 模型加载成功: {vad_path}")
        return _vad_model
    except Exception as e:
        logger.warning(f"fsmn-vad 加载失败: {e}")
        return None


def _get_campplus_model():
    """延迟加载 cam++ 模型"""
    global _campplus_model
    if _campplus_model is not None:
        return _campplus_model

    os.environ["MODELSCOPE_OFFLINE"] = "1"
    from funasr import AutoModel

    campplus_path = _MODELS_DIR / "speech_campplus_sv_zh-cn_16k-common"
    if not campplus_path.is_dir():
        logger.warning(f"cam++ 模型未找到: {campplus_path}")
        return None

    try:
        _campplus_model = AutoModel(model=str(campplus_path), disable_update=True)
        logger.info(f"cam++ 模型加载成功: {campplus_path}")
        return _campplus_model
    except Exception as e:
        logger.warning(f"cam++ 加载失败: {e}")
        return None


# ─────────────────────────────────────────
# VAD：使用 fsmn-vad 神经网络
# ─────────────────────────────────────────

def neural_vad(audio: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
    """使用 fsmn-vad 模型进行语音活动检测，返回 (start_sec, end_sec) 列表"""
    vad_model = _get_vad_model()
    if vad_model is None:
        # 回退到简单能量 VAD
        return energy_vad_fallback(audio, sr)

    try:
        res = vad_model.generate(input=audio)
        # fsmn-vad 返回格式: [{"value": [[start_ms, end_ms], [start_ms, end_ms], ...]}]
        if not res or not isinstance(res, list) or len(res) == 0:
            return [(0.0, len(audio) / sr)]

        item = res[0]
        if isinstance(item, dict) and "value" in item:
            intervals = item["value"]
            regions = []
            for interval in intervals:
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    start_ms, end_ms = interval[0], interval[1]
                    regions.append((start_ms / 1000.0, end_ms / 1000.0))
            if regions:
                return regions

        return [(0.0, len(audio) / sr)]
    except Exception as e:
        logger.warning(f"fsmn-vad 失败: {e}，回退能量VAD")
        return energy_vad_fallback(audio, sr)


def energy_vad_fallback(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 30,
    min_speech_ms: int = 300,
    min_silence_ms: int = 500,
) -> list[tuple[float, float]]:
    """后备：简单能量 VAD"""
    frame_len = int(sr * frame_ms / 1000)
    if len(audio) < frame_len:
        return [(0.0, len(audio) / sr)] if len(audio) > 0 else []

    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))

    median_rms = float(np.median(rms))
    threshold = max(0.005, median_rms * 1.5)
    voiced = rms > threshold

    # 提取连续语音段
    regions = []
    in_speech = False
    start_frame = 0
    for i, v in enumerate(voiced):
        if v and not in_speech:
            start_frame = i
            in_speech = True
        elif not v and in_speech:
            in_speech = False
            start_sec = start_frame * frame_ms / 1000
            end_sec = i * frame_ms / 1000
            if (end_sec - start_sec) * 1000 >= min_speech_ms:
                regions.append((start_sec, end_sec))
    if in_speech:
        start_sec = start_frame * frame_ms / 1000
        end_sec = n_frames * frame_ms / 1000
        if (end_sec - start_sec) * 1000 >= min_speech_ms:
            regions.append((start_sec, end_sec))

    return regions if regions else [(0.0, len(audio) / sr)]


# ─────────────────────────────────────────
# cam++ 嵌入提取
# ─────────────────────────────────────────

def _extract_campplus_embedding(model, audio_segment: np.ndarray) -> np.ndarray:
    """使用 cam++ 提取 192 维说话人嵌入"""
    res = model.generate(input=audio_segment)
    if res and isinstance(res, list) and len(res) > 0:
        item = res[0]
        if isinstance(item, dict) and "spk_embedding" in item:
            emb = item["spk_embedding"]
            if hasattr(emb, "numpy"):
                emb = emb.numpy()
            return np.array(emb).flatten()
    return np.zeros(192)


# ─────────────────────────────────────────
# 聚类
# ─────────────────────────────────────────

def cluster_speakers(
    embeddings: list[np.ndarray],
    n_speakers: int = 0,
    max_speakers: int = 6,
) -> list[int]:
    """对嵌入向量聚类，返回每段的说话人标签"""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n <= 1:
        return [0] * n

    X = np.array(embeddings)

    if n_speakers > 0:
        k = min(n_speakers, n)
        labels = AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(X)
        return labels.tolist()

    # 自动选择最佳簇数
    best_score = -1.0
    best_labels = [0] * n

    for k in range(2, min(max_speakers + 1, n + 1)):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_labels = labels.tolist()
        except Exception:
            continue

    # 分数太低认为是单人
    if best_score < 0.1:
        return [0] * n

    return best_labels


# ─────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────

def diarize(
    audio: np.ndarray,
    sr: int = 16000,
    n_speakers: int = 0,
    max_segment_seconds: float = 20.0,
) -> list[Segment]:
    """
    完整说话人分离流程：
    fsmn-vad 神经网络分段 → 切段 → cam++ 嵌入 → 聚类

    Returns: 带 speaker 标签的 Segment 列表
    """
    # 1. 神经网络 VAD
    regions = neural_vad(audio, sr=sr)
    logger.info(f"VAD 检测到 {len(regions)} 个语音段")

    if not regions:
        regions = [(0.0, len(audio) / sr)]

    # 2. 切段（过长的段切分）
    segments: list[Segment] = []
    for start, end in regions:
        duration = end - start
        if duration > max_segment_seconds:
            n_parts = int(np.ceil(duration / max_segment_seconds))
            part_len = duration / n_parts
            for i in range(n_parts):
                s = start + i * part_len
                e = start + (i + 1) * part_len
                s_idx = int(s * sr)
                e_idx = min(int(e * sr), len(audio))
                seg_audio = audio[s_idx:e_idx]
                if len(seg_audio) > sr * 0.3:
                    segments.append(Segment(start=s, end=e, audio=seg_audio))
        else:
            s_idx = int(start * sr)
            e_idx = min(int(end * sr), len(audio))
            seg_audio = audio[s_idx:e_idx]
            if len(seg_audio) > sr * 0.3:
                segments.append(Segment(start=start, end=end, audio=seg_audio))

    if not segments:
        return []

    logger.info(f"切分为 {len(segments)} 段待识别")

    # 3. 提取说话人嵌入
    campplus = _get_campplus_model()
    embeddings = []
    for seg in segments:
        if campplus is not None:
            emb = _extract_campplus_embedding(campplus, seg.audio)
        else:
            # 后备 MFCC
            import librosa
            mfcc = librosa.feature.mfcc(y=seg.audio, sr=sr, n_mfcc=20)
            emb = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        embeddings.append(emb)

    # 4. 聚类
    if len(segments) > 1:
        labels = cluster_speakers(embeddings, n_speakers=n_speakers)
        for seg, label in zip(segments, labels):
            seg.speaker = label
    else:
        segments[0].speaker = 0

    n_spk = len(set(s.speaker for s in segments))
    method = "cam++" if campplus else "MFCC"
    vad_method = "fsmn-vad" if _vad_model else "energy"
    logger.info(f"说话人分离完成: {len(segments)} 段, {n_spk} 位说话人 (VAD={vad_method}, 嵌入={method})")
    return segments


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """加载音频文件为 float32 单声道"""
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)
