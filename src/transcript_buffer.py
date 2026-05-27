"""
转写结果数据结构与缓冲区
"""
import threading
from dataclasses import dataclass


@dataclass
class ASRResult:
    """单段识别结果"""
    text: str
    timestamp: float = 0.0    # 距录音开始的秒数
    duration: float = 0.0     # 段时长（秒）
    speaker: int = -1         # 说话人编号，-1 表示未知
    is_partial: bool = False  # 是否为流式中间结果


class TranscriptBuffer:
    """线程安全的转写文本缓冲区"""

    def __init__(self):
        self._lock = threading.Lock()
        self._segments: list[ASRResult] = []
        self._partial: str = ""
        self._version: int = 0

    def append(self, result: ASRResult) -> None:
        if not result.text or not result.text.strip():
            return
        with self._lock:
            if result.is_partial:
                self._partial = result.text.strip()
            else:
                self._segments.append(result)
                self._partial = ""
            self._version += 1

    def get_display_text(self) -> str:
        """返回 Markdown 友好的显示文本"""
        with self._lock:
            lines = []
            for seg in self._segments:
                ts_min = int(seg.timestamp) // 60
                ts_sec = int(seg.timestamp) % 60
                ts_str = f"[{ts_min:02d}:{ts_sec:02d}]"
                spk_str = f" 说话人{seg.speaker}" if seg.speaker >= 0 else ""
                lines.append(f"{ts_str}{spk_str} {seg.text}")
            if self._partial:
                lines.append(f"... {self._partial}")
            return "\n".join(lines)

    def get_full_text(self) -> str:
        """返回拼接后的纯文本，供 LLM 总结"""
        with self._lock:
            parts = []
            for seg in self._segments:
                spk_str = f"[说话人{seg.speaker}] " if seg.speaker >= 0 else ""
                parts.append(f"{spk_str}{seg.text}")
            return "\n".join(parts)

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._partial = ""
            self._version = 0
