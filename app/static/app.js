"use strict";
const $ = (id) => document.getElementById(id);
const state = { job: null, es: null, poll: null, tick: null };
const PROCESSING_STATES = new Set([
  "queued", "downloading", "analyzing", "scoring",
  "render_queued", "rendering", "cancelling",
]);

const STAGE_LABELS = [
  ["download", "Download"], ["transcribe", "Transcribe"], ["scenes", "Scenes"],
  ["faces", "Face tracking"], ["scoring", "Viral scoring"],
];

init();

async function init() {
  try {
    const hw = await getJSON("/api/hardware");
    $("hw").textContent =
      `${hw.cpu_threads} threads · ${hw.encoder} · whisper ${hw.whisper_model}` +
      ` · ${hw.render_workers} render workers · scoring: ${hw.scoring_provider}` +
      (hw.has_cuda ? " · CUDA" : "");
    if (hw.auth_enabled) $("logoutLink").classList.remove("hidden");
  } catch { $("hw").textContent = "backend offline?"; }
  $("logoutLink").addEventListener("click", async (e) => {
    e.preventDefault();
    try { await postJSON("/api/logout", {}); } catch { /* cookie may already be gone */ }
    location.href = "/login";
  });
  $("serverRestartBtn").addEventListener("click", onServerRestart);
  $("serverStopBtn").addEventListener("click", onServerStop);
  $("urlForm").addEventListener("submit", onSubmit);
  $("selAll").addEventListener("click", (e) => { e.preventDefault(); setAll(true); });
  $("selNone").addEventListener("click", (e) => { e.preventDefault(); setAll(false); });
  $("renderBtn").addEventListener("click", onRender);
  $("stopBtn").addEventListener("click", onStop);
  $("clearCacheBtn").addEventListener("click", onClearCache);
  refreshHistory();
  initSettings();
}

async function onStop() {
  if (!state.job) return;
  if (!confirm("Stop processing this video? Any work currently in progress will be discarded.")) return;
  $("stopBtn").disabled = true;
  try { await postJSON(`/api/jobs/${state.job.id}/cancel`, {}); }
  catch (err) { showAlert(err.message); $("stopBtn").disabled = false; }
}

async function onServerRestart() {
  const btn = $("serverRestartBtn");
  if (!confirm(
    "Restart the Clipping Tool server? Any video currently processing will be stopped."
  )) return;
  btn.disabled = true;
  $("serverStopBtn").disabled = true;
  btn.textContent = "Restarting server…";
  try {
    const result = await postJSON("/api/server/restart", {});
    disconnect();
    $("hw").textContent = result.cancelled_jobs.length
      ? `server restarting · cancelled ${result.cancelled_jobs.length} video job(s)`
      : "server restarting…";
    await waitForServer();
    location.reload();
  } catch (err) {
    showAlert(err.message);
    btn.disabled = false;
    $("serverStopBtn").disabled = false;
    btn.textContent = "Restart server";
  }
}

async function waitForServer() {
  // The old process may answer briefly before execv replaces it. First wait
  // for that handover window, then poll until the fresh server is ready.
  await new Promise((resolve) => setTimeout(resolve, 1400));
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      const response = await fetch("/api/hardware", { cache: "no-store" });
      if (response.ok) return;
    } catch { /* server is between processes */ }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Server restart is taking longer than expected. Refresh this page in a moment.");
}

async function onServerStop() {
  const btn = $("serverStopBtn");
  if (!confirm(
    "Stop the whole Clipping Tool server? Any running video processing will also stop. " +
    "You will need to start the server again from the terminal or hosting dashboard."
  )) return;
  btn.disabled = true;
  btn.textContent = "Stopping server…";
  try {
    const result = await postJSON("/api/server/stop", {});
    disconnect();
    $("hw").textContent = result.cancelled_jobs.length
      ? `server stopping · cancelled ${result.cancelled_jobs.length} video job(s)`
      : "server stopping…";
  } catch (err) {
    showAlert(err.message);
    btn.disabled = false;
    btn.textContent = "Stop server";
  }
}

async function onClearCache() {
  if (!confirm("Delete the cached source videos + intermediates of all finished jobs? Your rendered clips are kept.")) return;
  const btn = $("clearCacheBtn");
  btn.disabled = true;
  try {
    const r = await postJSON("/api/maintenance/clear-caches", {});
    showAlert(`Cleared ${r.cleared} cache(s), freed ${r.freed_gb} GB.`);
    refreshHistory();
  } catch (err) { showAlert(err.message); }
  finally { btn.disabled = false; }
}

async function onSubmit(e) {
  e.preventDefault();
  hideAlert();
  $("goBtn").disabled = true;
  try {
    const r = await postJSON("/api/jobs", { url: $("urlInput").value.trim() });
    watch(r.job_id);
  } catch (err) { showAlert(err.message); }
  finally { $("goBtn").disabled = false; }
}

function watch(jobId) {
  disconnect();
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;
  es.onmessage = (ev) => render(JSON.parse(ev.data));
  es.onerror = () => {          // SSE dropped: fall back to polling
    es.close();
    if (state.poll) clearInterval(state.poll);
    state.poll = setInterval(async () => {
      try {
        const job = await getJSON(`/api/jobs/${jobId}`);
        render(job);
        if (isTerminal(job.state)) { clearInterval(state.poll); state.poll = null; }
      } catch { /* keep trying */ }
    }, 2000);
  };
}

function disconnect() {
  if (state.es) { state.es.close(); state.es = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  if (state.tick) { clearInterval(state.tick); state.tick = null; }
}

const isTerminal = (s) => ["done", "done_with_errors", "error", "cancelled"].includes(s);

function render(job) {
  state.job = job;
  $("progressPanel").classList.remove("hidden");
  $("jobTitle").textContent = job.meta?.title || job.url;
  renderSteps(job);
  startElapsed(job);

  // Only show the video stop control while actual processing is active.
  // awaiting_approval is deliberately excluded: no worker is running there.
  const stopBtn = $("stopBtn");
  stopBtn.classList.toggle("hidden", !PROCESSING_STATES.has(job.state));
  stopBtn.disabled = job.state === "cancelling";
  stopBtn.textContent = job.state === "cancelling"
    ? "Stopping video…"
    : "Stop video processing";

  const showCands = job.state === "awaiting_approval";
  $("candidatesPanel").classList.toggle("hidden", !showCands);
  if (showCands) renderCandidates(job);

  const showGallery = isTerminal(job.state) && (job.clips?.length || job.state !== "done");
  $("galleryPanel").classList.toggle("hidden", !showGallery && !isTerminal(job.state));
  if (isTerminal(job.state)) { renderGallery(job); disconnectIfDone(job); refreshHistory(); }
}

function disconnectIfDone(job) { if (isTerminal(job.state)) disconnect(); }

function renderSteps(job) {
  const steps = [];
  for (const [key, label] of STAGE_LABELS) {
    const pct = job.stages[key] === "running" && job.progress?.[key] != null
      ? ` ${Math.round(job.progress[key] * 100)}%` : "";
    steps.push(stepChip(label + pct, job.stages[key], job.timings[key]));
  }
  if (job.state === "awaiting_approval") steps.push(stepChip("Waiting for your picks", "running"));
  for (const [k, v] of Object.entries(job.stages)) {
    if (k.startsWith("render_")) steps.push(stepChip(`Clip ${parseInt(k.slice(7))}`, v));
  }
  if (isTerminal(job.state)) steps.push(stepChip(job.state.replaceAll("_", " "), job.state === "error" ? "failed" : "done"));
  $("steps").innerHTML = steps.join("");
}

function stepChip(label, status, secs) {
  const cls = ["running", "done", "failed", "review"].includes(status) ? status : "";
  const t = secs ? `<span class="t">${secs}s</span>` : "";
  return `<div class="step ${cls}"><span class="dot"></span>${esc(label)}${t}</div>`;
}

function startElapsed(job) {
  if (state.tick) clearInterval(state.tick);
  if (isTerminal(job.state)) { $("elapsed").textContent = ""; return; }
  const t0 = job.created * 1000;
  const upd = () => {
    const s = Math.max(0, (Date.now() - t0) / 1000);
    $("elapsed").textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")} elapsed`;
  };
  upd();
  state.tick = setInterval(upd, 1000);
}

function renderCandidates(job) {
  if (!job.candidates.length) {
    $("candidates").innerHTML = `<div class="minor">No strong moments found in this video.</div>`;
    $("renderBtn").disabled = true;
    return;
  }
  $("candidates").innerHTML = job.candidates.map((c) => {
    const cls = c.score >= 75 ? "hi" : c.score >= 50 ? "mid" : "lo";
    const checked = c.score >= 60 ? "checked" : "";
    return `<label class="cand">
      <input type="checkbox" data-rank="${c.rank}" ${checked}>
      <div class="body">
        <div class="title">#${c.rank} ${esc(c.title)}</div>
        <div class="meta">${hms(c.start)}–${hms(c.end)} · ${Math.round(c.duration)}s · scored by ${esc(c.provider)}</div>
        <div class="hook">“${esc(c.hook)}”</div>
        <div class="why">${esc(c.reason)}</div>
      </div>
      <span class="score ${cls}">${Math.round(c.score)}</span>
    </label>`;
  }).join("");
  $("candidates").querySelectorAll("input").forEach((cb) => cb.addEventListener("change", updateRenderBtn));
  updateRenderBtn();
}

function selectedRanks() {
  return [...$("candidates").querySelectorAll("input:checked")].map((cb) => +cb.dataset.rank);
}
function setAll(on) {
  $("candidates").querySelectorAll("input").forEach((cb) => (cb.checked = on));
  updateRenderBtn();
}
function updateRenderBtn() {
  const n = selectedRanks().length;
  $("renderBtn").textContent = `Render selected (${n})`;
  $("renderBtn").disabled = n === 0;
}

async function onRender() {
  const ranks = selectedRanks();
  if (!ranks.length || !state.job) return;
  $("renderBtn").disabled = true;
  try {
    await postJSON(`/api/jobs/${state.job.id}/approve`, { ranks });
    $("candidatesPanel").classList.add("hidden");
  } catch (err) { showAlert(err.message); $("renderBtn").disabled = false; }
}

function renderGallery(job) {
  $("galleryPanel").classList.remove("hidden");
  $("outNote").textContent = job.meta?.slug ? `saved to output/${job.meta.slug}/` : "";
  const ga = $("galleryAlert");
  if (job.error) {
    ga.innerHTML = `${esc(job.error)} <button id="retryBtn" style="margin-left:12px">Retry</button>`;
    ga.classList.remove("hidden");
    document.getElementById("retryBtn").addEventListener("click", async () => {
      try {
        await postJSON(`/api/jobs/${job.id}/retry`, {});
        ga.classList.add("hidden");
        $("galleryPanel").classList.add("hidden");
        watch(job.id);
      } catch (err) { showAlert(err.message); }
    });
  } else ga.classList.add("hidden");
  $("gallery").innerHTML = (job.clips || []).map((c) => {
    const qaBad = c.qa && !c.qa.passed;
    const qaLine = qaBad
      ? `<div class="qa-warn">⚠ needs review: ${esc((c.qa.flags || []).join("; "))}</div>` : "";
    return `<div class="clip${qaBad ? " flagged" : ""}">
      <video controls preload="metadata" src="${c.url}"></video>
      <div class="name">${esc(c.file)} · ${c.duration}s</div>
      ${qaLine}
      <a href="${c.url}" download>Download</a>
    </div>`;
  }).join("") || `<div class="minor">no clips rendered</div>`;
}

async function refreshHistory() {
  try {
    const jobsList = await getJSON("/api/jobs");
    if (!jobsList.length) return;
    $("history").innerHTML = jobsList.map((j) => {
      const processing = PROCESSING_STATES.has(j.state);
      const cls = j.state.startsWith("done") ? "done"
        : (j.state === "error" || j.state === "cancelled") ? "error" : "active";
      const d = new Date(j.created * 1000).toLocaleString();
      const del = isTerminal(j.state)
        ? `<button class="del ghost" data-del="${j.id}" title="Delete this job's cache + clips">✕</button>` : "";
      const stop = processing
        ? `<button class="hist-stop danger" data-stop="${j.id}">Stop processing</button>` : "";
      return `<div class="hist" data-id="${j.id}">
        <span>${esc(j.meta?.title || j.url)}</span>
        <span class="hist-actions"><span class="badge ${cls}">${esc(j.state)}</span>
          <span class="minor">${d}</span> ${stop} ${del}</span>
      </div>`;
    }).join("");
    $("history").querySelectorAll(".hist").forEach((el) =>
      el.addEventListener("click", async (ev) => {
        if (ev.target.closest("[data-del], [data-stop]")) return;
        const job = await getJSON(`/api/jobs/${el.dataset.id}`);
        render(job);
        if (!isTerminal(job.state)) watch(job.id);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }));
    $("history").querySelectorAll("[data-stop]").forEach((btn) =>
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm("Stop processing this video? Current work will be discarded.")) return;
        btn.disabled = true;
        btn.textContent = "Stopping…";
        try {
          await postJSON(`/api/jobs/${btn.dataset.stop}/cancel`, {});
          if (state.job?.id === btn.dataset.stop) $("stopBtn").disabled = true;
          await refreshHistory();
        } catch (err) {
          showAlert(err.message);
          btn.disabled = false;
          btn.textContent = "Stop processing";
        }
      }));
    $("history").querySelectorAll("[data-del]").forEach((btn) =>
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm("Delete this job — its cached files AND rendered clips?")) return;
        try { await fetch(`/api/jobs/${btn.dataset.del}`, { method: "DELETE" }); refreshHistory(); }
        catch (err) { showAlert(err.message); }
      }));

    // A browser refresh must reconnect to an already-running job. Without
    // this, the progress panel (and its Stop button) stayed hidden.
    const active = jobsList.find((job) => PROCESSING_STATES.has(job.state));
    if (!state.job && active) watch(active.id);
  } catch { /* history is best-effort */ }
}

// settings
async function initSettings() {
  state.clearedKeys = new Set();
  await refreshSettingsView();
  document.querySelectorAll(".clearKey").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const input = $(a.dataset.target);
      input.value = "";
      state.clearedKeys.add(a.dataset.target);
      input.placeholder = "will be cleared on save";
    });
  });
  $("saveSettingsBtn").addEventListener("click", onSaveSettings);
}

async function refreshSettingsView() {
  try {
    const s = await getJSON("/api/settings");
    $("providerSelect").value = s.scoring_provider;
    $("openaiKeyInput").value = "";
    $("openaiKeyInput").placeholder = s.has_openai_key ? "•••••••• (saved — leave blank to keep)" : "not set";
    $("geminiKeyInput").value = "";
    $("geminiKeyInput").placeholder = s.has_gemini_key ? "•••••••• (saved — leave blank to keep)" : "not set";
  } catch { /* settings best-effort */ }
}

async function onSaveSettings() {
  const btn = $("saveSettingsBtn");
  const alertBox = $("settingsAlert");
  alertBox.classList.add("hidden");
  btn.disabled = true;
  const payload = { scoring_provider: $("providerSelect").value };
  for (const [inputId, field] of [["openaiKeyInput", "openai_api_key"], ["geminiKeyInput", "gemini_api_key"]]) {
    const val = $(inputId).value.trim();
    if (val) payload[field] = val;
    else if (state.clearedKeys.has(inputId)) payload[field] = "";
  }
  try {
    await postJSON("/api/settings", payload);
    state.clearedKeys.clear();
    await refreshSettingsView();
    alertBox.textContent = "Settings saved.";
    alertBox.classList.remove("hidden");
    alertBox.classList.add("good");
  } catch (err) {
    alertBox.textContent = err.message;
    alertBox.classList.remove("hidden", "good");
  } finally { btn.disabled = false; }
}

// helpers
function authRedirect(r) {
  // Session expired mid-use: bounce to the login page (no-op when auth is off).
  if (r.status === 401 && !location.pathname.startsWith("/login")) location.href = "/login";
}
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    authRedirect(r);
    throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  }
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    if (url !== "/api/logout") authRedirect(r);
    throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  }
  return r.json();
}
function hms(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  const h = Math.floor(m / 60);
  return h ? `${h}:${String(m % 60).padStart(2, "0")}:${String(s).padStart(2, "0")}`
           : `${m}:${String(s).padStart(2, "0")}`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function showAlert(msg) { $("alert").textContent = msg; $("alert").classList.remove("hidden"); }
function hideAlert() { $("alert").classList.add("hidden"); }
