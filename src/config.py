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
GRADIO_HOST: str = os.environ.get("GRADIO_HOST", "0.0.0.0")

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
# 文件上传：本地 FunASR 模型配置
# ─────────────────────────────────────────
# 模型目录（包含 paraformer/vad/punc/spk 四个模型子目录）
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR: str = os.environ.get(
    "MODELS_DIR",
    str(_SRC_DIR.parent / "models" / "iic")
)

# 并行处理配置（大文件拆分为 10 分钟块并行处理）
PARALLEL_WORKERS: int = int(os.environ.get("PARALLEL_WORKERS", "6"))
FILE_CHUNK_SECONDS: float = float(os.environ.get("FILE_CHUNK_SECONDS", "600.0"))  # 10分钟

# 内置热词（写死，不对外暴露）
_BUILTIN_HOTWORDS = "数字员工 智能体 MCP 飞书 昆仑小智 昆仑数智 昆仑智联 剑锋总 朝晖总"

# 用户可配置的额外热词（通过环境变量或前端传入）
ASR_CONTEXT: str = os.environ.get("ASR_CONTEXT", "")

# 合并后的完整热词（内置 + 用户配置）
def get_hotwords(user_hotwords: str = "") -> str:
    """合并内置热词和用户热词，返回空格分隔的完整列表"""
    parts = [_BUILTIN_HOTWORDS]
    if ASR_CONTEXT:
        parts.append(ASR_CONTEXT)
    if user_hotwords:
        parts.append(user_hotwords)
    return " ".join(parts)

# VAD 最大单段时长（毫秒）
ASR_VAD_MAX_SEGMENT_MS: int = int(os.environ.get("ASR_VAD_MAX_SEGMENT_MS", "60000"))

# SeACo 热词偏置权重（控制热词对识别结果的影响强度，默认1.0，越大热词越强）
SEACO_WEIGHT: float = float(os.environ.get("SEACO_WEIGHT", "1.0"))


# ─────────────────────────────────────────
# 飞书导出配置（云文档）
# ─────────────────────────────────────────
FEISHU_BASE_URL: str = os.environ.get(
    "FEISHU_BASE_URL", "https://open.fklzl.cnpc.com.cn"
)
FEISHU_APP_ID: str = os.environ.get(
    "FEISHU_APP_ID", "cli_aa8697c62eb8d366"
)
FEISHU_APP_SECRET: str = os.environ.get(
    "FEISHU_APP_SECRET", "wFErLj6okrtDUEBCKQCS8don4aiJIlEb"
)
FEISHU_DOC_BASE_URL: str = os.environ.get(
    "FEISHU_DOC_BASE_URL", "https://nipj5983sr.fklzl.cnpc.com.cn"
)


def validate() -> list[str]:
    """校验关键配置，返回警告信息列表"""
    warnings = []
    if not ASR_API_KEY:
        warnings.append("ASR_API_KEY 未配置，远程语音识别将不可用")
    if not LLM_API_KEY:
        warnings.append("LLM_API_KEY 未配置，会议纪要生成功能将不可用")
    return warnings
