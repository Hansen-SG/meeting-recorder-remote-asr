# 会议记录工具（双轨 ASR 架构）

实时录音使用远程 API，文件上传使用本地模型，两条链路各取所长。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│                         Web UI (浏览器)                        │
├────────────────────────────┬─────────────────────────────────┤
│   🎤 实时录音               │   📁 文件上传                    │
│   MediaRecorder 8秒切片     │   拖拽上传音频文件                │
│          ↓                 │          ↓                      │
│   远程 qwen3-asr-flash     │   本地 FunASR 一体化管线          │
│   /v1/chat/completions     │   paraformer-zh (ASR)           │
│   (多模态 audio 格式)       │   + fsmn-vad (语音检测)          │
│                            │   + ct-punc (标点恢复)           │
│                            │   + cam++ (说话人分离)            │
│                            │   + 热词支持                     │
├────────────────────────────┴─────────────────────────────────┤
│                    DeepSeek-v4-pro (LLM)                      │
│                    流式生成结构化会议纪要                        │
└──────────────────────────────────────────────────────────────┘
```

## 两种模式对比

| 特性 | 实时录音 | 文件上传 |
|------|---------|---------|
| ASR 引擎 | 远程 qwen3-asr-flash | 本地 paraformer-zh |
| 说话人分离 | 不支持 | cam++ 嵌入 + 聚类 |
| 热词识别 | 不支持（API 限制） | 支持（SeACo 机制） |
| 标点恢复 | 不支持 | ct-punc 自动标点 |
| 延迟 | ~10秒（8秒窗口 + 网络） | 取决于音频时长 |
| 依赖 | 仅需网络连接 | 需本地模型文件（~3GB） |

## 本地模型

文件上传模式需要以下模型（放在 `models/iic/` 目录下）：

| 模型 | 目录名 | 大小 | 用途 |
|------|--------|------|------|
| paraformer-zh | `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ~953MB | 语音识别 |
| fsmn-vad | `speech_fsmn_vad_zh-cn-16k-common-pytorch` | ~3.9MB | 语音活动检测 |
| ct-punc | `punc_ct-transformer_cn-en-common-vocab471067-large` | ~1.2GB | 标点恢复 |
| cam++ | `speech_campplus_sv_zh-cn_16k-common` | ~28MB | 说话人分离 |

## 配置

API 通过 `src/config.py` 配置（支持环境变量覆盖）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASR_API_BASE` | 远程 ASR 基地址 | `https://aiproxy.jla.petrotech.cnpc` |
| `ASR_API_KEY` | API Key | （内置） |
| `ASR_MODEL` | 远程模型名 | `qwen3-asr-flash` |
| `LLM_API_BASE` | LLM 基地址 | `https://aiproxy.jla.petrotech.cnpc/v1` |
| `LLM_MODEL` | LLM 模型名 | `deepseek-v4-pro` |
| `ASR_CONTEXT` | 热词列表（空格分隔） | `中石油 昆仑小智 昆仑数智 ...` |
| `ASR_VAD_MAX_SEGMENT_MS` | VAD 单段最大时长 | `60000` |
| `REALTIME_CHUNK_SECONDS` | 实时模式切片秒数 | `8.0` |

## 启动

```cmd
scripts\start.cmd
```

或手动：
```cmd
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install funasr modelscope
.venv\Scripts\python src\app.py
```

访问 http://127.0.0.1:7860

## 目录结构

```
├── src/
│   ├── app.py              # 入口（uvicorn 启动）
│   ├── server.py           # FastAPI 路由
│   ├── config.py           # 配置管理
│   ├── qwen_asr.py         # 远程 ASR 客户端（实时录音用）
│   ├── file_processor.py   # 文件上传处理（本地 FunASR 管线）
│   ├── speaker_diarize.py  # 说话人分离（fsmn-vad + cam++）
│   ├── llm_summary.py      # LLM 会议纪要生成
│   └── transcript_buffer.py# 转写缓冲区
├── web/
│   ├── index.html          # 前端页面
│   ├── app.js              # 前端逻辑
│   └── style.css           # 样式
├── scripts/start.cmd       # Windows 启动脚本
├── requirements.txt        # Python 依赖
└── models/iic/             # 本地模型（需单独下载，不含在 git 中）
```

## 网络要求

- 企业内网需设置 `NO_PROXY=.petrotech.cnpc`（代码已自动处理）
- SSL 证书验证已禁用（企业 CA 不在默认信任链）
