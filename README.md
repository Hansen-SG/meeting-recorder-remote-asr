# 远程 Qwen3-ASR 会议记录工具

基于远程 **qwen3-asr-flash** API 的会议记录工具，无需本地部署 ASR 模型。

## 功能

1. **实时录音转写**：从麦克风录音，每 8 秒批量调用远程 ASR，实时显示转写文本
2. **音频文件上传**：支持 mp3/wav/m4a/flac/ogg 等常见格式，自动说话人分离 + 远程识别
3. **自动会议纪要**：转写完成后调用 LLM 生成 Markdown 格式会议纪要

## 配置

API 通过环境变量配置（默认值已写在 `src/config.py`）：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `ASR_API_BASE` | Qwen3-ASR 远程 API 基地址 | `https://aiproxy.jla.petrotech.cnpc` |
| `ASR_API_KEY` | API Key | （内置） |
| `ASR_MODEL` | 模型名 | `qwen3-asr-flash` |
| `LLM_API_BASE` | LLM API 基地址 | `https://aiproxy.jla.petrotech.cnpc/v1` |
| `LLM_API_KEY` | LLM API Key | （内置） |
| `LLM_MODEL` | LLM 模型名 | `deepseek-v4-pro` |
| `GRADIO_PORT` | Web UI 端口 | `7860` |
| `REALTIME_CHUNK_SECONDS` | 实时模式批量秒数 | `8.0` |

## 本地启动

需要 Python 3.10+。

### 方式 1：脚本一键启动（推荐）

```cmd
scripts\start.cmd
```

会自动创建 `.venv` 并安装依赖。

### 方式 2：手动

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\app.py
```

启动后浏览器自动打开 http://127.0.0.1:7860

## 目录结构

```
remote-asr/
├── src/
│   ├── app.py              # 入口
│   ├── config.py           # 配置
│   ├── qwen_asr.py         # Qwen3-ASR 远程 API 客户端
│   ├── llm_summary.py      # LLM 纪要生成
│   ├── audio_capture.py    # 麦克风捕获
│   ├── realtime_worker.py  # 实时识别工作线程
│   ├── speaker_diarize.py  # 说话人分离（VAD+MFCC+聚类）
│   ├── file_processor.py   # 文件上传处理流程
│   ├── transcript_buffer.py# 转写结果缓冲区
│   └── web_ui.py           # Gradio UI
├── scripts/start.cmd       # Windows 启动脚本
├── requirements.txt
└── README.md
```

## 设计要点

- **实时识别**：因为远程 HTTP 调用有延迟，采用累积 8 秒后批量识别的策略，
  而非真正的流式 WebSocket。优点是无需复杂会话管理，缺点是延迟略高。
- **说话人分离**：纯本地实现（能量 VAD + MFCC + Agglomerative 聚类），
  不依赖 pyannote 等大型模型。准确度对会议场景一般可用。
- **API 风格自适应**：先尝试 DashScope 原生 multimodal-generation 接口，
  失败则回退到 OpenAI 兼容的 `/v1/audio/transcriptions`。

## 常见问题

**Q: 实时识别延迟高？**
A: 远程 API 调用本身有 1-3 秒延迟，加上 8 秒累积窗口，总延迟约 10 秒。
可以通过设置环境变量 `REALTIME_CHUNK_SECONDS=5` 降低累积时间。

**Q: 文件上传识别说话人不准？**
A: 默认是自动估计说话人数。如果已知人数，在 UI 上手动选择 1~6 可提高准确度。

**Q: 麦克风权限？**
A: 首次使用 Windows 会弹出隐私权限请求，需允许应用访问麦克风。
