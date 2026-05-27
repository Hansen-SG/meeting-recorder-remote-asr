"""
Qwen3-ASR-Flash 远程 API 客户端

调用方式：POST /v1/chat/completions（多模态格式）
音频以 base64 data URL 嵌入 message content 中：
  {"type": "audio", "audio": "data:audio/wav;base64,..."}

输入：音频文件路径 或 numpy 数组
输出：识别文本
"""
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

import config

logger = logging.getLogger(__name__)


class Qwen3ASRError(Exception):
    pass


def _np_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """将 numpy float32/int16 数组编码为 16-bit PCM WAV 字节流"""
    import wave

    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)
    elif audio.dtype == np.int16:
        audio_i16 = audio
    else:
        audio_i16 = audio.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_i16.tobytes())
    return buf.getvalue()


def _audio_to_data_url(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """编码为 base64 data url"""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _read_local_file(path: str) -> tuple[bytes, str]:
    """读取本地文件，返回（字节内容，MIME 类型）"""
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "aac": "audio/aac",
        "webm": "audio/webm",
    }
    mime = mime_map.get(suffix, "audio/wav")
    return p.read_bytes(), mime


class Qwen3ASRClient:
    """
    Qwen3-ASR-Flash 远程 API 客户端

    通过 /v1/chat/completions 多模态接口调用 ASR 模型。
    """

    def __init__(
        self,
        api_key: str = config.ASR_API_KEY,
        api_base: str = config.ASR_API_BASE,
        model: str = config.ASR_MODEL,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ─────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = "zh",
        context: Optional[str] = None,
    ) -> dict:
        """识别本地音频文件（任意格式）"""
        audio_bytes, mime = _read_local_file(audio_path)
        return self._transcribe_bytes(audio_bytes, mime, language, context)

    def transcribe_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "zh",
        context: Optional[str] = None,
    ) -> dict:
        """识别 numpy 音频数组（已经是单声道）"""
        wav_bytes = _np_to_wav_bytes(audio, sample_rate)
        return self._transcribe_bytes(wav_bytes, "audio/wav", language, context)

    # ─────────────────────────────────────────
    # 内部
    # ─────────────────────────────────────────

    def _transcribe_bytes(
        self,
        audio_bytes: bytes,
        mime: str,
        language: Optional[str],
        context: Optional[str],
    ) -> dict:
        """通过 /v1/chat/completions 多模态接口调用 ASR"""
        url = f"{self.api_base}/v1/chat/completions"
        data_url = _audio_to_data_url(audio_bytes, mime)

        # 注意：qwen3-asr-flash 不支持 system message，仅 user message 中嵌入音频
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": data_url},
            ],
        }]

        payload = {
            "model": self.model,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            t0 = time.time()
            resp = requests.post(
                url, json=payload, headers=headers,
                timeout=self.timeout, verify=False
            )
            elapsed = time.time() - t0

            if resp.status_code >= 400:
                raise Qwen3ASRError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )

            logger.info(f"[Qwen3ASR] 识别成功，耗时 {elapsed:.1f}s")
            return self._parse_response(resp.json())

        except requests.RequestException as e:
            raise Qwen3ASRError(f"请求失败: {e}")

    def _parse_response(self, data: dict) -> dict:
        """解析 chat completions 响应"""
        choices = data.get("choices") or []
        if not choices:
            raise Qwen3ASRError(f"响应无 choices: {json.dumps(data)[:500]}")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""

        # 即使是空文本也返回（纯噪音/静音时正常）
        return {"text": text.strip(), "raw": data}
