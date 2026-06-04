/**
 * Record Worker - 在独立线程中处理音频切片和上传
 * 不受主线程/后台 Tab 影响，保证持续工作
 *
 * 职责：
 * 1. 接收 AudioWorklet 传来的 PCM float32 数据
 * 2. 按时间窗口（chunkSecs）积累并切片
 * 3. 重采样到 16kHz（如果源采样率不同）
 * 4. 编码为 WAV 并 POST 到 /api/transcribe-chunk
 * 5. 将识别结果回传给主线程
 */

let config = {
  chunkSecs: 8,
  hotwords: '',
  serverUrl: '/api/transcribe-chunk',
};

let buffer = [];       // Float32Array 片段列表
let bufferSamples = 0; // 当前积累的采样数
let sourceSampleRate = 48000;
let chunkIndex = 0;
let startTime = 0;
let active = false;

self.onmessage = function (e) {
  const msg = e.data;

  switch (msg.type) {
    case 'start':
      config.chunkSecs = msg.chunkSecs || 8;
      config.hotwords = msg.hotwords || '';
      config.serverUrl = msg.serverUrl || '/api/transcribe-chunk';
      sourceSampleRate = msg.sampleRate || 48000;
      buffer = [];
      bufferSamples = 0;
      chunkIndex = 0;
      startTime = msg.startTime || Date.now();
      active = true;
      break;

    case 'audio':
      if (!active) break;
      const samples = new Float32Array(msg.samples);
      buffer.push(samples);
      bufferSamples += samples.length;

      // 检查是否达到切片时长
      const targetSamples = config.chunkSecs * sourceSampleRate;
      if (bufferSamples >= targetSamples) {
        flushChunk();
      }
      break;

    case 'stop':
      active = false;
      // 发送最后残余数据
      if (bufferSamples > sourceSampleRate * 0.5) { // 至少 0.5 秒
        flushChunk();
      }
      break;

    case 'updateConfig':
      if (msg.chunkSecs) config.chunkSecs = msg.chunkSecs;
      if (msg.hotwords !== undefined) config.hotwords = msg.hotwords;
      break;
  }
};

function flushChunk() {
  if (bufferSamples === 0) return;

  // 合并缓冲区
  const merged = new Float32Array(bufferSamples);
  let offset = 0;
  for (const buf of buffer) {
    merged.set(buf, offset);
    offset += buf.length;
  }
  buffer = [];
  bufferSamples = 0;

  // 重采样到 16kHz
  const resampled = resample(merged, sourceSampleRate, 16000);

  // 静音检测
  let rms = 0;
  for (let i = 0; i < resampled.length; i++) {
    rms += resampled[i] * resampled[i];
  }
  rms = Math.sqrt(rms / resampled.length);
  if (rms < 0.002) {
    // 纯静音，跳过
    self.postMessage({ type: 'skipped', reason: 'silence' });
    return;
  }

  // 编码为 16-bit PCM WAV
  const wavBytes = encodeWav(resampled, 16000);

  // 计算时间戳
  const timestamp = (Date.now() - startTime) / 1000 - config.chunkSecs;

  // 上传
  const idx = chunkIndex++;
  uploadChunk(wavBytes, timestamp, idx);
}

async function uploadChunk(wavBytes, timestamp, idx) {
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  const form = new FormData();
  form.append('audio', blob, `chunk-${idx}.wav`);
  form.append('language', 'zh');
  form.append('timestamp', String(Math.max(0, timestamp)));
  form.append('hotwords', config.hotwords);

  try {
    const resp = await fetch(config.serverUrl, { method: 'POST', body: form });
    if (!resp.ok) {
      self.postMessage({ type: 'error', message: `HTTP ${resp.status}`, idx });
      return;
    }
    const data = await resp.json();
    const text = (data.text || '').trim();
    if (text) {
      self.postMessage({ type: 'result', text, timestamp: Math.max(0, timestamp), idx });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message, idx });
  }
}

/**
 * 简单线性插值重采样
 */
function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.round(input.length / ratio);
  const output = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const idx0 = Math.floor(srcIdx);
    const idx1 = Math.min(idx0 + 1, input.length - 1);
    const frac = srcIdx - idx0;
    output[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
  }
  return output;
}

/**
 * 编码为 16-bit PCM WAV
 */
function encodeWav(samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = samples.length * blockAlign;
  const headerSize = 44;
  const totalSize = headerSize + dataSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, totalSize - 8, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);           // chunk size
  view.setUint16(20, 1, true);            // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // PCM samples (float32 -> int16)
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s * 0x7FFF, true);
    offset += 2;
  }

  return buffer;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
