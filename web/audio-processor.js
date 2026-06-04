/**
 * AudioWorklet Processor - 在音频渲染线程中运行
 * 不受浏览器后台 Tab 节流影响，持续采集 PCM 数据
 * 通过 port.postMessage 将音频数据发送给主线程/Worker
 */
class RecordProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffers = [];
    this._sampleCount = 0;
    this._active = true;

    this.port.onmessage = (e) => {
      if (e.data.command === 'stop') {
        this._active = false;
      }
    };
  }

  process(inputs, outputs, parameters) {
    if (!this._active) return false;

    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // 取第一个通道（单声道）
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // 复制数据（AudioWorklet 中的 buffer 是短暂的）
    const copy = new Float32Array(channelData.length);
    copy.set(channelData);

    this._buffers.push(copy);
    this._sampleCount += copy.length;

    // 每积累约 0.5 秒就发送一批（减少消息频率）
    // sampleRate 通常是 44100 或 48000，0.5s ≈ 22050 samples
    if (this._sampleCount >= sampleRate * 0.5) {
      // 合并缓冲区
      const total = new Float32Array(this._sampleCount);
      let offset = 0;
      for (const buf of this._buffers) {
        total.set(buf, offset);
        offset += buf.length;
      }
      // 发送给主线程（transferable 避免拷贝）
      this.port.postMessage(
        { type: 'audio', samples: total.buffer, sampleRate: sampleRate },
        [total.buffer]
      );
      this._buffers = [];
      this._sampleCount = 0;
    }

    return true;
  }
}

registerProcessor('record-processor', RecordProcessor);
