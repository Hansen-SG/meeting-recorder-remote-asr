"""
配置管理模块
从环境变量读取所有配置，提供合理默认值
"""
import os

# ─────────────────────────────────────────
# 网络环境修复：企业代理绕过 + SSL 证书
# ─────────────────────────────────────────
# 企业内网 API 域名需绕过代理
_NO_PROXY_DOMAINS = ".petrotech.cnpc"
for key in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(key, "")
    if _NO_PROXY_DOMAINS not in existing:
        os.environ[key] = f"{existing},{_NO_PROXY_DOMAINS}" if existing else _NO_PROXY_DOMAINS

# 禁用 SSL 验证警告（企业 CA 不在默认信任链）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# requests 全局禁用 SSL 验证
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""


# ─────────────────────────────────────────
# Qwen3-ASR-Flash 远程 API 配置
# ─────────────────────────────────────────
ASR_API_BASE: str = os.environ.get(
    "ASR_API_BASE", "https://aiproxy.jla.petrotech.cnpc"
)
ASR_API_KEY: str = os.environ.get(
    "ASR_API_KEY", "sk-C8xKcnNYUAWoZ4M6LGc9PuIXLuhbnbPoj9G45GV35u2fEdTb"
)
ASR_MODEL: str = os.environ.get("ASR_MODEL", "qwen3-asr-flash")

# ─────────────────────────────────────────
# LLM 会议纪要 API 配置（沿用原系统）
# ─────────────────────────────────────────
LLM_API_BASE: str = os.environ.get(
    "LLM_API_BASE", "https://aiproxy.jla.petrotech.cnpc/v1"
)
LLM_API_KEY: str = os.environ.get(
    "LLM_API_KEY", "sk-C8xKcnNYUAWoZ4M6LGc9PuIXLuhbnbPoj9G45GV35u2fEdTb"
)
LLM_MODEL: str = os.environ.get("LLM_MODEL", "deepseek-v4-pro")

# ─────────────────────────────────────────
# Web UI 配置
# ─────────────────────────────────────────
GRADIO_PORT: int = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_HOST: str = os.environ.get("GRADIO_HOST", "127.0.0.1")

# ─────────────────────────────────────────
# 音频配置
# ─────────────────────────────────────────
SAMPLE_RATE: int = 16000           # 16kHz 采样率
CHANNELS: int = 1                  # 单声道
CHUNK_SIZE: int = 1600             # 100ms 的采样数 (16000 * 0.1)
AUDIO_QUEUE_MAXSIZE: int = 200     # 音频队列最大长度

# 实时识别：每多少秒的音频送一次远程识别（远程API有HTTP延迟，太短不划算）
REALTIME_CHUNK_SECONDS: float = float(os.environ.get("REALTIME_CHUNK_SECONDS", "8.0"))
# 静音检测阈值（float32 RMS）：低于此值视为静音段跳过
SILENCE_THRESHOLD: float = float(os.environ.get("SILENCE_THRESHOLD", "0.002"))

# ─────────────────────────────────────────
# 说话人分离配置（用于文件上传场景）
# ─────────────────────────────────────────
# 使用简单的 MFCC + KMeans 聚类做说话人分离
# 默认聚类数量 (None=自动估计 1~6)
SPEAKER_CLUSTERS: int = int(os.environ.get("SPEAKER_CLUSTERS", "0"))  # 0=自动
# 单段最长（秒），切分后送ASR
DIARIZE_SEGMENT_SECONDS: float = float(os.environ.get("DIARIZE_SEGMENT_SECONDS", "20.0"))

# 热词/上下文（提升专有名词识别）
ASR_CONTEXT: str = os.environ.get(
    "ASR_CONTEXT",
    "中石油 中国石油 集团公司 数字员工 智能体 会议纪要 飞书 大模型 昆仑小智 昆仑数智",
)

# VAD 最大单段时长（毫秒）
ASR_VAD_MAX_SEGMENT_MS: int = int(os.environ.get("ASR_VAD_MAX_SEGMENT_MS", "60000"))


def validate() -> list[str]:
    """校验关键配置，返回警告信息列表"""
    warnings = []
    if not ASR_API_KEY:
        warnings.append("ASR_API_KEY 未配置，远程语音识别将不可用")
    if not LLM_API_KEY:
        warnings.append("LLM_API_KEY 未配置，会议纪要生成功能将不可用")
    return warnings
