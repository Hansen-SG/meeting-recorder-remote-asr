// 远程 Qwen3-ASR 会议记录 - 前端
// 实时录音通过浏览器 MediaRecorder，按时间窗口切片上传到后端

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
  // 实时录音
  // ─────────────────────────────────────
  const RT = {
    mediaStream: null,
    recorder: null,
    chunks: [],
    segments: [],          // {ts, text}
    partial: "",
    startTime: 0,
    chunkIndex: 0,
    timerId: null,
    audioCtx: null,
    analyser: null,
    vuId: null,
    cycleId: null,
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

  // 启动 VU meter
  function startVuMeter(stream) {
    try {
      RT.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = RT.audioCtx.createMediaStreamSource(stream);
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
    if (RT.audioCtx) {
      try { RT.audioCtx.close(); } catch {}
      RT.audioCtx = null;
    }
    vuBar.style.width = "0%";
  }

  // 选用浏览器支持的 mime
  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const m of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    }
    return "";
  }

  async function startRecording() {
    if (RT.isRunning) return;
    try {
      RT.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (e) {
      setRtStatus("❌ 无法访问麦克风：" + e.message);
      return;
    }

    RT.segments = [];
    RT.partial = "";
    RT.chunkIndex = 0;
    RT.startTime = Date.now();
    RT.isRunning = true;
    renderRtTranscript();

    btnStart.disabled = true;
    btnStop.disabled = false;
    setRtStatus("🔴 录音中…", true);

    startVuMeter(RT.mediaStream);

    // 启动 1Hz 定时刷新
    RT.timerId = setInterval(() => {
      const sec = (Date.now() - RT.startTime) / 1000;
      rtTimer.textContent = fmtTime(sec);
    }, 500);

    // 周期重启 MediaRecorder，每个周期是一个独立完整的音频片段
    cycleRecorder();
  }

  function cycleRecorder() {
    if (!RT.isRunning) return;

    const chunkSecs = Math.max(3, Math.min(30, Number(chunkSecsInput.value) || 8));
    const mime = pickMimeType();
    const opts = mime ? { mimeType: mime } : {};

    let recorder;
    try {
      recorder = new MediaRecorder(RT.mediaStream, opts);
    } catch (e) {
      setRtStatus("❌ MediaRecorder 创建失败: " + e.message);
      stopRecording();
      return;
    }

    const localChunks = [];
    const chunkStartTs = (Date.now() - RT.startTime) / 1000;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) localChunks.push(e.data);
    };

    recorder.onstop = async () => {
      const blob = new Blob(localChunks, { type: mime || "audio/webm" });
      if (blob.size > 1000) {
        sendChunkForRecognition(blob, chunkStartTs, mime);
      }
      // 启动下一个周期
      if (RT.isRunning) {
        // 立即开始下一段，避免漏录
        cycleRecorder();
      }
    };

    recorder.start();
    RT.recorder = recorder;
    RT.cycleId = setTimeout(() => {
      try {
        if (recorder.state !== "inactive") recorder.stop();
      } catch {}
    }, chunkSecs * 1000);
  }

  async function sendChunkForRecognition(blob, timestamp, mime) {
    const form = new FormData();
    const ext = (mime || "").includes("mp4") ? ".m4a" :
                (mime || "").includes("ogg") ? ".ogg" : ".webm";
    form.append("audio", blob, `chunk-${RT.chunkIndex++}${ext}`);
    form.append("language", "zh");
    form.append("timestamp", String(timestamp));

    try {
      const r = await fetch("/api/transcribe-chunk", { method: "POST", body: form });
      if (!r.ok) {
        const errTxt = await r.text();
        console.warn("识别失败:", errTxt);
        return;
      }
      const data = await r.json();
      const text = (data.text || "").trim();
      if (text) {
        RT.segments.push({ ts: timestamp, text });
        renderRtTranscript();
      }
    } catch (e) {
      console.warn("识别请求异常", e);
    }
  }

  async function stopRecording() {
    if (!RT.isRunning) return;
    RT.isRunning = false;

    if (RT.cycleId) { clearTimeout(RT.cycleId); RT.cycleId = null; }
    if (RT.timerId) { clearInterval(RT.timerId); RT.timerId = null; }

    // 停止当前段，让 onstop 把最后一段送上去
    if (RT.recorder && RT.recorder.state !== "inactive") {
      try { RT.recorder.stop(); } catch {}
    }
    if (RT.mediaStream) {
      RT.mediaStream.getTracks().forEach((t) => t.stop());
      RT.mediaStream = null;
    }
    stopVuMeter();

    btnStart.disabled = false;
    btnStop.disabled = true;
    setRtStatus(`✅ 已停止（${RT.segments.length} 句）`);

    if (autoFlushInput.checked && RT.segments.length > 0) {
      // 等 2 秒让最后一段识别完
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
  }

  btnUpload.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnUpload.disabled = true;
    btnSummaryUp.disabled = true;
    upTranscript.innerHTML = "";
    upSummary.innerHTML = "";
    uploadProgress.style.width = "5%";
    uploadStatus.textContent = "上传文件中…";

    const form = new FormData();
    form.append("audio", selectedFile);
    form.append("n_speakers", speakersSelect.value);

    try {
      const r = await fetch("/api/transcribe-file", { method: "POST", body: form });
      if (!r.ok) throw new Error("HTTP " + r.status);

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      uploadProgress.style.width = "30%";

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
    }
  });

  function handleUploadEvent(obj) {
    if (obj.event === "start") {
      uploadStatus.textContent = `处理中: ${obj.filename}`;
      uploadProgress.style.width = "40%";
    } else if (obj.event === "transcript") {
      uploadProgress.style.width = "90%";
      lastLlmInput = obj.llm_input || "";
      renderUploadTranscript(obj.text || "");
      const s = obj.stats || {};
      uploadStatus.textContent =
        `✅ 完成：时长 ${(s.total_duration || 0).toFixed(1)}s, ` +
        `${s.recognized || 0}/${s.segments || 0} 段, ${s.speakers || 0} 位说话人`;
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
    } catch (e) {
      upSummary.innerHTML += `<p style="color: var(--danger)">❌ ${escapeHtml(e.message)}</p>`;
    } finally {
      btnSummaryUp.disabled = false;
      btnSummaryUp.textContent = "重新生成";
    }
  });
})();
