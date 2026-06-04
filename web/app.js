// 远程 Qwen3-ASR 会议记录 - 前端
// 实时录音通过浏览器 MediaRecorder，按时间窗口切片上传到后端

// 全局 401 拦截：未登录时跳转登录页
const _origFetch = window.fetch;
window.fetch = async function(...args) {
  const resp = await _origFetch.apply(this, args);
  if (resp.status === 401) {
    window.location.href = "/auth/login";
  }
  return resp;
};

(() => {
  // ─────────────────────────────────────
  // 通用工具
  // ─────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const fmtTime = (sec) => {
    sec = Math.max(0, Math.floor(sec));
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  };

  // 极简 Markdown 渲染器（标题/列表/表格/粗体/代码）
  function renderMarkdown(md) {
    if (!md) return "";
    md = md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // 代码块
    md = md.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c}</code></pre>`);

    // 标题
    md = md.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    md = md.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    md = md.replace(/^# (.*)$/gm, "<h1>$1</h1>");

    // 表格（GFM）
    md = md.replace(
      /((?:^\|.*\|\n)+)/gm,
      (block) => {
        const rows = block.trim().split("\n");
        if (rows.length < 2) return block;
        const headers = rows[0].split("|").slice(1, -1).map((s) => s.trim());
        // 跳过分隔行 rows[1]
        const dataRows = rows.slice(2);
        let html = "<table><thead><tr>";
        headers.forEach((h) => (html += `<th>${h}</th>`));
        html += "</tr></thead><tbody>";
        dataRows.forEach((r) => {
          const cells = r.split("|").slice(1, -1).map((s) => s.trim());
          html += "<tr>";
          cells.forEach((c) => (html += `<td>${c}</td>`));
          html += "</tr>";
        });
        html += "</tbody></table>";
        return html;
      }
    );

    // 列表
    md = md.replace(/^(?:- |\* )(.*)$/gm, "<li>$1</li>");
    md = md.replace(/(<li>[\s\S]*?<\/li>(?:\n<li>[\s\S]*?<\/li>)*)/g, "<ul>$1</ul>");

    // 内联
    md = md.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    md = md.replace(/`([^`]+)`/g, "<code>$1</code>");

    // 段落（剩余的非空非块级行）
    md = md
      .split(/\n\n+/)
      .map((p) => {
        const t = p.trim();
        if (!t) return "";
        if (/^<(h\d|ul|ol|pre|table|li)/.test(t)) return t;
        return `<p>${t.replace(/\n/g, "<br/>")}</p>`;
      })
      .join("\n");

    return md;
  }

  // 安全转 HTML
  const escapeHtml = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // ─────────────────────────────────────
  // Tab 切换
  // ─────────────────────────────────────
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      $$(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#panel-${tab.dataset.tab}`).classList.add("active");
    })
  );

  // ─────────────────────────────────────
  // 健康检查
  // ─────────────────────────────────────
  async function healthCheck() {
    const pill = $("#health-pill");
    try {
      const r = await fetch("/api/health");
      if (!r.ok) throw new Error(r.status);
      const data = await r.json();
      pill.textContent = `ASR ${data.asr_model} · LLM ${data.llm_model}`;
      pill.classList.add("ok");
      $("#footer-info").textContent =
        `ASR @ ${data.asr_base}  ·  LLM @ ${data.llm_base}`;
    } catch (e) {
      pill.textContent = "服务不可用";
      pill.classList.add("err");
    }
  }
  healthCheck();

  // ─────────────────────────────────────
  // 用户信息显示
  // ─────────────────────────────────────
  async function loadUserInfo() {
    try {
      const r = await fetch("/auth/user");
      if (!r.ok) return;
      const data = await r.json();
      if (data.logged_in && data.name) {
        const el = $("#user-info");
        $("#user-name").textContent = data.name;
        el.style.display = "flex";
      }
    } catch (e) { /* ignore */ }
  }
  loadUserInfo();

  // ─────────────────────────────────────
  // 热词管理
  // ─────────────────────────────────────
  const hotwordsInput = $("#hotwords-input");
  const uploadHotwordsInput = $("#upload-hotwords-input");

  async function loadHotwords() {
    try {
      const r = await fetch("/api/hotwords");
      if (r.ok) {
        const data = await r.json();
        const hw = data.hotwords || "";
        hotwordsInput.value = hw;
        hotwordsInput.placeholder = "输入热词，空格分隔，如：中石油 大模型 昆仑数智";
        uploadHotwordsInput.value = hw;
        uploadHotwordsInput.placeholder = "输入热词，空格分隔，如：中石油 大模型 昆仑数智";
      }
    } catch (e) {
      hotwordsInput.placeholder = "输入热词，空格分隔";
      uploadHotwordsInput.placeholder = "输入热词，空格分隔";
    }
  }
  loadHotwords();

  // ─────────────────────────────────────
  // 实时录音（AudioWorklet + Web Worker 方案）
  // 后台 Tab 不受节流影响，持续录音
  // ─────────────────────────────────────
  const RT = {
    mediaStream: null,
    sysStream: null,
    audioCtx: null,
    workletNode: null,
    worker: null,
    segments: [],          // {ts, text}
    partial: "",
    startTime: 0,
    timerId: null,
    analyser: null,
    vuId: null,
    isRunning: false,
  };

  const btnStart = $("#btn-start");
  const btnStop = $("#btn-stop");
  const rtStatus = $("#rt-status");
  const rtTranscript = $("#rt-transcript");
  const rtTimer = $("#rt-timer");
  const vuBar = $(".vu-bar");
  const chunkSecsInput = $("#chunk-secs");
  const autoFlushInput = $("#auto-flush");

  function renderRtTranscript() {
    const parts = RT.segments.map((s) => {
      return `<div class="line"><span class="ts">[${fmtTime(s.ts)}]</span>${escapeHtml(s.text)}</div>`;
    });
    if (RT.partial) {
      parts.push(`<div class="line partial"><span class="ts">…</span>${escapeHtml(RT.partial)}</div>`);
    }
    rtTranscript.innerHTML = parts.join("");
    rtTranscript.scrollTop = rtTranscript.scrollHeight;
  }

  function setRtStatus(msg, recording = false) {
    rtStatus.textContent = msg;
    if (recording) {
      btnStart.classList.add("recording");
    } else {
      btnStart.classList.remove("recording");
    }
  }

  // VU meter（使用同一个 AudioContext）
  function startVuMeter() {
    try {
      if (!RT.audioCtx || !RT.mediaStream) return;
      const src = RT.audioCtx.createMediaStreamSource(RT.mediaStream);
      RT.analyser = RT.audioCtx.createAnalyser();
      RT.analyser.fftSize = 512;
      src.connect(RT.analyser);
      const data = new Uint8Array(RT.analyser.frequencyBinCount);
      const tick = () => {
        if (!RT.analyser) return;
        RT.analyser.getByteTimeDomainData(data);
        let max = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.abs(data[i] - 128) / 128;
          if (v > max) max = v;
        }
        vuBar.style.width = Math.min(100, max * 200) + "%";
        RT.vuId = requestAnimationFrame(tick);
      };
      tick();
    } catch (e) {
      console.warn("VU meter init failed", e);
    }
  }

  function stopVuMeter() {
    if (RT.vuId) cancelAnimationFrame(RT.vuId);
    RT.vuId = null;
    RT.analyser = null;
    vuBar.style.width = "0%";
  }

  async function startRecording() {
    if (RT.isRunning) return;

    // 1. 获取麦克风
    let micStream = null;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (e) {
      setRtStatus("❌ 无法访问麦克风：" + e.message);
      return;
    }

    // 2. 尝试获取系统音频（通过屏幕共享的音频轨道）
    let sysStream = null;
    try {
      sysStream = await navigator.mediaDevices.getDisplayMedia({
        video: { width: 1, height: 1, frameRate: 1 },  // 最小视频（必须请求）
        audio: true,  // 系统音频
      });
      // 关闭不需要的视频轨道
      sysStream.getVideoTracks().forEach((t) => t.stop());
    } catch (e) {
      // 用户拒绝或浏览器不支持系统音频，仅用麦克风继续
      console.info("系统音频不可用（仅录制麦克风）:", e.message);
    }

    // 3. 创建 AudioContext + 混合音频源
    try {
      RT.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await RT.audioCtx.audioWorklet.addModule("/static/audio-processor.js");
      RT.workletNode = new AudioWorkletNode(RT.audioCtx, "record-processor");

      // 混合麦克风 + 系统音频（如果有）
      const micSource = RT.audioCtx.createMediaStreamSource(micStream);
      micSource.connect(RT.workletNode);

      if (sysStream && sysStream.getAudioTracks().length > 0) {
        const sysSource = RT.audioCtx.createMediaStreamSource(sysStream);
        sysSource.connect(RT.workletNode);
      }

      // 保存 stream 引用以便后续关闭
      RT.mediaStream = micStream;
      RT.sysStream = sysStream;
    } catch (e) {
      setRtStatus("❌ AudioWorklet 初始化失败：" + e.message);
      micStream.getTracks().forEach((t) => t.stop());
      if (sysStream) sysStream.getTracks().forEach((t) => t.stop());
      return;
    }

    // 3. 创建 Web Worker
    RT.worker = new Worker("/static/record-worker.js");
    RT.worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "result") {
        RT.segments.push({ ts: msg.timestamp, text: msg.text });
        renderRtTranscript();
      } else if (msg.type === "error") {
        console.warn("Worker 识别错误:", msg.message);
      }
    };

    // 4. AudioWorklet → Worker 数据桥接
    RT.workletNode.port.onmessage = (e) => {
      if (e.data.type === "audio" && RT.worker) {
        RT.worker.postMessage(
          { type: "audio", samples: e.data.samples, sampleRate: e.data.sampleRate },
          [e.data.samples]
        );
      }
    };

    // 5. 启动 Worker
    const chunkSecs = Math.max(3, Math.min(30, Number(chunkSecsInput.value) || 8));
    RT.segments = [];
    RT.partial = "";
    RT.startTime = Date.now();
    RT.isRunning = true;

    RT.worker.postMessage({
      type: "start",
      chunkSecs: chunkSecs,
      hotwords: (hotwordsInput.value || "").trim(),
      sampleRate: RT.audioCtx.sampleRate,
      startTime: RT.startTime,
    });

    renderRtTranscript();
    btnStart.disabled = true;
    btnStop.disabled = false;
    const hasSys = RT.sysStream && RT.sysStream.getAudioTracks().length > 0;
    setRtStatus(hasSys ? "🔴 录音中（麦克风+系统声音）" : "🔴 录音中（仅麦克风）", true);

    startVuMeter();

    // 计时器
    RT.timerId = setInterval(() => {
      const sec = (Date.now() - RT.startTime) / 1000;
      rtTimer.textContent = fmtTime(sec);
    }, 500);
  }

  async function stopRecording() {
    if (!RT.isRunning) return;
    RT.isRunning = false;

    // 通知 Worker 停止并发送残余数据
    if (RT.worker) {
      RT.worker.postMessage({ type: "stop" });
      // 给 Worker 2 秒处理最后一段
      setTimeout(() => {
        if (RT.worker) { RT.worker.terminate(); RT.worker = null; }
      }, 3000);
    }

    // 停止 AudioWorklet
    if (RT.workletNode) {
      RT.workletNode.port.postMessage({ command: "stop" });
      RT.workletNode.disconnect();
      RT.workletNode = null;
    }

    // 关闭 AudioContext
    if (RT.audioCtx) {
      try { await RT.audioCtx.close(); } catch {}
      RT.audioCtx = null;
    }

    // 停止麦克风
    if (RT.mediaStream) {
      RT.mediaStream.getTracks().forEach((t) => t.stop());
      RT.mediaStream = null;
    }
    // 停止系统音频
    if (RT.sysStream) {
      RT.sysStream.getTracks().forEach((t) => t.stop());
      RT.sysStream = null;
    }

    stopVuMeter();
    if (RT.timerId) { clearInterval(RT.timerId); RT.timerId = null; }

    btnStart.disabled = false;
    btnStop.disabled = true;
    setRtStatus(`✅ 已停止（${RT.segments.length} 句）`);

    if (autoFlushInput.checked && RT.segments.length > 0) {
      setTimeout(() => generateRtSummary(), 2500);
    }
  }

  btnStart.addEventListener("click", startRecording);
  btnStop.addEventListener("click", stopRecording);

  // ─────────────────────────────────────
  // 实时纪要生成（SSE）
  // ─────────────────────────────────────
  const btnSummaryRt = $("#btn-summary-rt");
  const rtSummary = $("#rt-summary");

  function buildRtTranscriptText() {
    return RT.segments.map((s) => s.text).join("\n");
  }

  async function generateRtSummary() {
    const text = buildRtTranscriptText();
    if (text.trim().length < 20) {
      rtSummary.innerHTML = `<p style="color: var(--text-soft)">转写内容太短，无法生成纪要。</p>`;
      return;
    }
    btnSummaryRt.disabled = true;
    btnSummaryRt.textContent = "生成中…";
    rtSummary.innerHTML = `<p style="color: var(--text-soft)">⏳ 正在生成纪要…</p>`;

    let acc = "";
    try {
      await streamSummary(text, (token) => {
        acc += token;
        rtSummary.innerHTML = renderMarkdown(acc);
      });
    } catch (e) {
      rtSummary.innerHTML += `<p style="color: var(--danger)">❌ ${escapeHtml(e.message)}</p>`;
    } finally {
      btnSummaryRt.disabled = false;
      btnSummaryRt.textContent = "重新生成";
    }
  }

  btnSummaryRt.addEventListener("click", generateRtSummary);

  // 实时录音导出按钮
  const btnExportRt = $("#btn-export-rt");
  btnExportRt.addEventListener("click", async () => {
    const transcript = buildRtTranscriptText();
    const summary = rtSummary.innerText || "";
    await exportToDoc(transcript, summary, btnExportRt);
  });

  // 纪要生成完毕后启用导出按钮
  const _origGenerateRtSummary = generateRtSummary;
  generateRtSummary = async function () {
    await _origGenerateRtSummary();
    if (rtSummary.innerText.trim().length > 10) {
      btnExportRt.disabled = false;
    }
  };

  // 通用导出函数
  async function exportToDoc(transcript, summary, btn) {
    btn.disabled = true;
    btn.textContent = "导出中…";
    try {
      const form = new FormData();
      form.append("transcript", transcript);
      form.append("summary", summary);
      const r = await fetch("/api/export", { method: "POST", body: form });
      const data = await r.json();
      if (data.ok && data.url) {
        btn.textContent = "导出成功";
        window.open(data.url, "_blank");
      } else {
        btn.textContent = "导出失败";
        alert("导出失败: " + (data.error || "未知错误"));
      }
    } catch (e) {
      btn.textContent = "导出失败";
      alert("导出异常: " + e.message);
    } finally {
      setTimeout(() => {
        btn.disabled = false;
        btn.textContent = "📤 导出云文档";
      }, 3000);
    }
  }

  // 通用 SSE 流式接收
  async function streamSummary(transcript, onToken, title = null) {
    const form = new FormData();
    form.append("transcript", transcript);
    if (title) form.append("title", title);

    const r = await fetch("/api/summary", { method: "POST", body: form });
    if (!r.ok) throw new Error("HTTP " + r.status);

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        try {
          const obj = JSON.parse(payload);
          if (obj.event === "token") onToken(obj.text);
          else if (obj.event === "error") throw new Error(obj.message);
          else if (obj.event === "done") return;
        } catch (e) {
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }
  }

  // ─────────────────────────────────────
  // 文件上传
  // ─────────────────────────────────────
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const dzFilename = $("#dz-filename");
  const btnUpload = $("#btn-upload");
  const speakersSelect = $("#speakers-select");
  const uploadStatus = $("#upload-status");
  const uploadProgress = $("#upload-progress");
  const upTranscript = $("#up-transcript");
  const upSummary = $("#up-summary");
  const btnSummaryUp = $("#btn-summary-up");

  let selectedFile = null;
  let lastLlmInput = "";

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) onFileSelected(fileInput.files[0]);
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  });

  function onFileSelected(file) {
    selectedFile = file;
    dzFilename.textContent = `已选择: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    btnUpload.disabled = false;
    // 选择文件后立即查询一次队列状态
    fetchQueueStatus();
  }

  // ─────────────────────────────────────
  // 队列状态轮询（每20秒）
  // ─────────────────────────────────────
  const queueInfoWrap = $("#queue-info-wrap");
  const queueInfo = $("#queue-info");
  const queueText = $("#queue-text");
  let queuePollTimer = null;

  async function fetchQueueStatus() {
    try {
      const r = await fetch("/api/queue-status");
      if (!r.ok) return;
      const data = await r.json();
      updateQueueDisplay(data);
    } catch (e) {
      // 网络失败时隐藏
      queueInfoWrap.style.display = "none";
    }
  }

  function updateQueueDisplay(data) {
    const { processing, waiting, total } = data;
    queueInfoWrap.style.display = "block";

    if (total === 0) {
      queueInfo.className = "queue-info idle";
      queueText.textContent = "当前无排队，上传后可立即处理";
    } else {
      queueInfo.className = "queue-info";
      const pos = waiting + 1; // 自己将是队列中的下一个
      const estMinutes = Math.ceil(processing * 2 + waiting * 2); // 粗略估算
      let msg = `当前 ${processing} 个任务正在处理`;
      if (waiting > 0) msg += `，${waiting} 个排队中`;
      msg += `。您上传后预计第 ${pos} 位`;
      if (estMinutes > 0) msg += `，约等 ${estMinutes} 分钟`;
      queueText.textContent = msg;
    }
  }

  function startQueuePolling() {
    stopQueuePolling();
    fetchQueueStatus();
    queuePollTimer = setInterval(fetchQueueStatus, 20000); // 每20秒
  }

  function stopQueuePolling() {
    if (queuePollTimer) {
      clearInterval(queuePollTimer);
      queuePollTimer = null;
    }
  }

  // 页面加载时开始轮询
  startQueuePolling();

  btnUpload.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnUpload.disabled = true;
    btnSummaryUp.disabled = true;
    upTranscript.innerHTML = "";
    upSummary.innerHTML = "";
    uploadProgress.style.width = "2%";
    const sizeMB = (selectedFile.size / 1024 / 1024).toFixed(1);
    uploadStatus.textContent = `上传中 (${sizeMB} MB)…`;

    // 隐藏队列提示，进入处理状态
    queueInfoWrap.style.display = "none";
    stopQueuePolling();

    const form = new FormData();
    form.append("audio", selectedFile);
    form.append("n_speakers", speakersSelect.value);
    form.append("hotwords", (uploadHotwordsInput.value || "").trim());

    try {
      const r = await fetch("/api/transcribe-file", { method: "POST", body: form });
      if (!r.ok) throw new Error("HTTP " + r.status);

      // 上传完成，开始接收 SSE 流
      uploadProgress.style.width = "10%";
      uploadStatus.textContent = "已上传，正在识别…";

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split("\n\n");
        buf = events.pop();
        for (const ev of events) {
          const line = ev.trim();
          if (!line.startsWith("data:")) continue;
          try {
            const obj = JSON.parse(line.slice(5).trim());
            handleUploadEvent(obj);
          } catch {}
        }
      }
    } catch (e) {
      uploadStatus.textContent = "❌ 失败: " + e.message;
      uploadProgress.style.width = "0%";
    } finally {
      btnUpload.disabled = false;
      // 处理完成后恢复轮询
      startQueuePolling();
    }
  });

  function handleUploadEvent(obj) {
    if (obj.event === "start") {
      uploadStatus.textContent = `处理中: ${obj.filename}`;
      uploadProgress.style.width = "10%";
    } else if (obj.event === "progress") {
      const pct = 10 + (obj.done / Math.max(obj.total, 1)) * 75;
      uploadProgress.style.width = pct + "%";
      uploadStatus.textContent = obj.msg || `处理中 ${obj.done}/${obj.total}`;
    } else if (obj.event === "transcript") {
      uploadProgress.style.width = "90%";
      lastLlmInput = obj.llm_input || "";
      renderUploadTranscript(obj.text || "");
      const s = obj.stats || {};
      const dur = s.total_duration || 0;
      const durStr = dur >= 60 ? `${(dur/60).toFixed(1)}分钟` : `${dur.toFixed(1)}秒`;
      uploadStatus.textContent =
        `✅ 完成：音频 ${durStr}, ` +
        `${s.recognized || 0} 段, ${s.speakers || 0} 位说话人`;
      btnSummaryUp.disabled = false;
    } else if (obj.event === "done") {
      uploadProgress.style.width = "100%";
      setTimeout(() => (uploadProgress.style.width = "0%"), 1500);
    } else if (obj.event === "error") {
      uploadStatus.textContent = "❌ " + obj.message;
      uploadProgress.style.width = "0%";
    }
  }

  function renderUploadTranscript(text) {
    // 行格式：[mm:ss] 说话人X 文本
    const re = /^\[(\d{2}):(\d{2})\](?:\s+说话人(\d+))?\s*(.*)$/;
    const html = text.split("\n").map((line) => {
      const m = re.exec(line);
      if (!m) return `<div class="line">${escapeHtml(line)}</div>`;
      const ts = `[${m[1]}:${m[2]}]`;
      const spk = m[3];
      const txt = m[4] || "";
      const spkClass = spk ? `spk spk${parseInt(spk) % 6}` : "";
      const spkHtml = spk ? `<span class="${spkClass}">说话人 ${spk}</span>` : "";
      return `<div class="line"><span class="ts">${ts}</span>${spkHtml}${escapeHtml(txt)}</div>`;
    }).join("");
    upTranscript.innerHTML = html;
  }

  btnSummaryUp.addEventListener("click", async () => {
    if (!lastLlmInput) return;
    btnSummaryUp.disabled = true;
    btnSummaryUp.textContent = "生成中…";
    upSummary.innerHTML = `<p style="color: var(--text-soft)">⏳ 正在生成纪要…</p>`;
    let acc = "";
    try {
      await streamSummary(lastLlmInput, (tok) => {
        acc += tok;
        upSummary.innerHTML = renderMarkdown(acc);
      });
      // 纪要生成完毕，启用导出按钮
      btnExportUp.disabled = false;
    } catch (e) {
      upSummary.innerHTML += `<p style="color: var(--danger)">❌ ${escapeHtml(e.message)}</p>`;
    } finally {
      btnSummaryUp.disabled = false;
      btnSummaryUp.textContent = "重新生成";
    }
  });

  // 上传音频导出按钮
  const btnExportUp = $("#btn-export-up");
  btnExportUp.addEventListener("click", async () => {
    const transcript = upTranscript.innerText || "";
    const summary = upSummary.innerText || "";
    await exportToDoc(transcript, summary, btnExportUp);
  });
})();
