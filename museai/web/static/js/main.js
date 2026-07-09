(function () {
  "use strict";

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = new Intl.NumberFormat();
  const controllers = new Set();

  function postJSON(url, payload = {}) {
    const controller = new AbortController();
    controllers.add(controller);
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    }).finally(() => controllers.delete(controller));
  }

  window.addEventListener("beforeunload", () => {
    controllers.forEach((controller) => controller.abort());
  });

  function initDashboard() {
    const stream = document.getElementById("manuscript-stream");
    if (!stream) return;

    const wordCounter = document.getElementById("word-counter");
    const beatIdentifier = document.getElementById("beat-identifier");
    const pulse = document.getElementById("run-pulse");
    const runLabel = document.getElementById("run-status-label");
    const criticSummary = document.getElementById("critic-summary");
    const criticReasoning = document.getElementById("critic-reasoning");
    const criticTools = document.getElementById("critic-tools");
    const reviewPanel = document.getElementById("review-panel");
    const reviewText = document.getElementById("review-text");
    const reviewMeta = document.getElementById("review-meta");
    const padEmpty = document.getElementById("pad-empty");
    const controlResult = document.getElementById("control-result");
    let currentBeat = null;
    let autoScroll = true;
    const beatText = new Map();

    stream.addEventListener("scroll", () => {
      const distance = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
      autoScroll = distance < 80;
    });

    const radar = initRadar();

    function scrollIfNeeded() {
      if (autoScroll) stream.scrollTop = stream.scrollHeight;
    }

    function setPhase(phase) {
      document.querySelectorAll("[data-phase-step]").forEach((step) => {
        step.classList.toggle("active", step.dataset.phaseStep === phase);
      });
    }

    function setRunStatus(status) {
      const normalized = status || "idle";
      if (runLabel) runLabel.textContent = normalized[0].toUpperCase() + normalized.slice(1);
      if (!pulse) return;
      pulse.className = "pulse-dot";
      if (normalized === "running" || normalized === "done" || normalized === "review") pulse.classList.add("pulse-running");
      else if (normalized === "paused") pulse.classList.add("pulse-paused");
      else if (normalized === "stopped") pulse.classList.add("pulse-stopped");
      else pulse.classList.add("pulse-idle");
    }

    function renderMarkdown(text) {
      if (window.marked) return window.marked.parse(text || "");
      return (text || "").replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
    }

    function ensureBeat(beatId, label) {
      const id = beatId || "unknown-beat";
      let block = stream.querySelector(`[data-beat-id="${CSS.escape(id)}"]`);
      if (!block) {
        stream.querySelector(".empty-state")?.remove();
        block = document.createElement("div");
        block.className = "beat-block";
        block.dataset.beatId = id;
        block.innerHTML = `<div class="beat-label">${label || id}</div><div class="beat-prose"></div>`;
        stream.appendChild(block);
      }
      currentBeat = id;
      return block;
    }

    const handlers = {
      hydration(snapshot) {
        Object.entries(snapshot || {}).forEach(([type, data]) => dispatch(type, data));
      },
      run_status(data) {
        setRunStatus(data.status);
        if (data.status === "idle") setPhase("Idle");
      },
      phase_change(data) {
        setPhase(data.phase);
      },
      chapters_planned(data) {
        if (beatIdentifier) beatIdentifier.textContent = `${data.arc_id || "Arc"} · Ch ${data.active_chapter_id || "—"} · Beat —`;
      },
      beats_planned(data) {
        const active = (data.beats || []).find((beat) => beat.id === data.active_beat_id) || (data.beats || [])[0];
        if (beatIdentifier && active) beatIdentifier.textContent = `${data.chapter_id || "Chapter"} · Beat ${active.ordering}`;
      },
      pad_update(data) {
        const pad = data.target_pad || {};
        radar.data.datasets[0].data = [pad.pleasure || 0, pad.arousal || 0, pad.dominance || 0];
        radar.update();
        if (padEmpty) padEmpty.textContent = `${data.character_id || "Character"} · ${data.beat_id || "beat"}`;
      },
      beat_start(data) {
        const label = `Beat ${data.ordering || "—"} · ${data.beat_id || "unknown"}`;
        ensureBeat(data.beat_id, label);
        beatText.set(data.beat_id, "");
        scrollIfNeeded();
      },
      token(data) {
        const beatId = data.beat_id || currentBeat || "unknown-beat";
        const block = ensureBeat(beatId, beatId);
        const next = (beatText.get(beatId) || "") + (data.text || "");
        beatText.set(beatId, next);
        block.querySelector(".beat-prose").innerHTML = renderMarkdown(next);
        scrollIfNeeded();
      },
      audit(data) {
        if (criticSummary) criticSummary.textContent = data.failures?.length ? `${data.failures.length} audit issue(s)` : "Programmatic audit clean";
      },
      critic_tool(data) {
        if (!criticTools) return;
        const args = data.arguments || {};
        const results = Array.isArray(data.result) ? data.result : [];
        const preview = results.slice(0, 2).map((item) => `${item.title || "Result"}: ${item.snippet || item.url || ""}`).join("\n");
        const card = document.createElement("div");
        card.className = "critic-tool";
        card.innerHTML = `<strong>${data.tool || "web_search"}</strong><div class="text-muted small">${args.query || "No query"}</div><pre class="small mb-0">${preview || "No results returned."}</pre>`;
        criticTools.prepend(card);
      },
      critic_reasoning(data) {
        if (criticReasoning) criticReasoning.textContent = data.text || "Critic returned no text.";
      },
      critic_summary(data) {
        if (criticSummary) criticSummary.textContent = data.summary || "Critic complete";
      },
      revision(data) {
        const block = ensureBeat(data.beat_id, `Revised ${data.beat_id}`);
        beatText.set(data.beat_id, data.text || "");
        block.querySelector(".beat-prose").innerHTML = renderMarkdown(data.text || "");
      },
      word_count(data) {
        if (wordCounter) wordCounter.textContent = `${fmt.format(data.word_count || 0)} / ${fmt.format(data.target || 0)} words`;
      },
      pointer_update(data) {
        const pointer = data.fsm_pointer || {};
        if (beatIdentifier) beatIdentifier.textContent = `Arc ${pointer.arc_id || "—"} · Ch ${pointer.chapter_id || "—"} · Beat ${(pointer.beat_index ?? "—")}`;
      },
      review_needed(data) {
        reviewPanel?.classList.remove("d-none");
        if (reviewText) reviewText.value = data.best_seen_draft || "";
        if (reviewMeta) reviewMeta.textContent = `${(data.failures || []).length} issue(s); best failure count ${data.best_seen_failure_count ?? "unknown"}.`;
      },
      manuscript_ready(data) {
        setRunStatus("done");
        if (criticSummary) criticSummary.textContent = `Manuscript ready: ${data.path || "export complete"}`;
      },
    };

    function dispatch(type, data) {
      const handler = handlers[type];
      if (handler) handler(data || {});
    }

    function addSse(type) {
      events.addEventListener(type, (event) => dispatch(type, JSON.parse(event.data || "{}")));
    }

    const events = new EventSource("/stream");
    Object.keys(handlers).forEach(addSse);

    document.getElementById("generate-button")?.addEventListener("click", async () => {
      const response = await postJSON("/generate");
      const body = await response.json();
      setRunStatus(body.status || "idle");
      if (!body.ok && criticSummary) criticSummary.textContent = body.error;
    });

    document.querySelectorAll("[data-control-action]").forEach((button) => {
      button.addEventListener("click", async () => {
        const action = button.dataset.controlAction;
        const response = await postJSON(`/control/${action}`);
        const body = await response.json();
        if (controlResult) controlResult.textContent = body.ok ? `Status: ${body.status}` : body.error;
        setRunStatus(body.status);
      });
    });

    document.getElementById("review-accept")?.addEventListener("click", () => submitReview("accept"));
    document.getElementById("review-regenerate")?.addEventListener("click", () => submitReview("regenerate"));

    async function submitReview(decision) {
      const response = await postJSON("/control/review", { decision, edited_text: reviewText?.value || "" });
      const body = await response.json();
      document.getElementById("review-result").textContent = body.ok ? `Review resolved: ${body.status}` : body.error;
      if (body.ok) reviewPanel?.classList.add("d-none");
    }
  }

  function initRadar() {
    const canvas = document.getElementById("pad-radar");
    if (!canvas || !window.Chart) return { data: { datasets: [{ data: [null, null, null] }] }, update() {} };
    const color = cssVar("--color-accent-info");
    return new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["Pleasure", "Arousal", "Dominance"],
        datasets: [{ label: "Target PAD", data: [null, null, null], borderColor: color, backgroundColor: "transparent" }],
      },
      options: { scales: { r: { min: -1, max: 1, ticks: { stepSize: 0.5 } } } },
    });
  }

  function initSettings() {
    const page = document.getElementById("settings-page");
    if (!page) return;
    const baseConfig = JSON.parse(page.dataset.config || "{}");
    const result = document.getElementById("settings-save-result");
    const endpointResult = document.getElementById("endpoint-test-result");

    function buildConfig() {
      const cfg = structuredClone(baseConfig);
      cfg.endpoint.base_url = document.getElementById("setting-base-url").value;
      cfg.endpoint.model_name = document.getElementById("setting-model-name").value;
      cfg.endpoint.api_key = document.getElementById("setting-api-key").value;
      cfg.endpoint.tokenizer_family = document.getElementById("setting-tokenizer").value;
      document.querySelectorAll(".generation-setting").forEach((input) => {
        const key = input.dataset.generationKey;
        const value = input.step === "1" || Number.isInteger(Number(input.value)) ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
        cfg.generation[key] = value;
      });
      return cfg;
    }

    document.getElementById("save-settings")?.addEventListener("click", async () => {
      const response = await postJSON("/settings/save", buildConfig());
      const body = await response.json();
      if (result) result.textContent = body.ok ? "Settings saved." : body.error;
    });

    document.getElementById("test-endpoint")?.addEventListener("click", async () => {
      if (endpointResult) endpointResult.textContent = "Testing…";
      const response = await postJSON("/settings/test_endpoint");
      const body = await response.json();
      if (endpointResult) endpointResult.textContent = body.ok ? `OK: ${body.model}` : body.error;
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
    initSettings();
  });
})();