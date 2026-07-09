(function () {
  "use strict";

  const fmt = new Intl.NumberFormat();
  const controllers = new Set();

  window.addEventListener("beforeunload", () => {
    controllers.forEach((controller) => controller.abort());
  });

  /** POST JSON and always resolve to an {ok, ...} object — never throws. */
  async function postJSON(url, payload = {}) {
    const controller = new AbortController();
    controllers.add(controller);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => null);
      if (!body || typeof body !== "object") {
        return { ok: false, error: `Unexpected response (HTTP ${response.status}).` };
      }
      return body;
    } catch (err) {
      if (err && err.name === "AbortError") return { ok: false, error: "Request cancelled.", aborted: true };
      return { ok: false, error: "Network error — is the MuseAI server running?" };
    } finally {
      controllers.delete(controller);
    }
  }

  async function getJSON(url) {
    try {
      const response = await fetch(url);
      const body = await response.json().catch(() => null);
      return body && typeof body === "object" ? body : null;
    } catch {
      return null;
    }
  }

  /** Render untrusted markdown (LLM output) into an element, sanitized. */
  function renderMarkdownInto(el, text) {
    const source = String(text || "");
    if (window.marked && window.DOMPurify) {
      el.innerHTML = window.DOMPurify.sanitize(window.marked.parse(source));
    } else {
      el.textContent = source;
    }
  }

  function setInlineResult(el, message, ok) {
    if (!el) return;
    el.textContent = message || "";
    el.classList.toggle("is-error", ok === false);
    el.classList.toggle("is-ok", ok === true);
  }

  /* ------------------------------------------------------------------ */
  /* Dashboard                                                           */
  /* ------------------------------------------------------------------ */

  const RUN_UI = {
    idle: { label: "Idle", pulse: "pulse-idle", generate: "Generate", pauseResume: null, stop: false },
    running: { label: "Running", pulse: "pulse-running", generate: null, pauseResume: "pause", stop: true },
    paused: { label: "Paused", pulse: "pulse-paused", generate: null, pauseResume: "resume", stop: true },
    review: { label: "Awaiting review", pulse: "pulse-review", generate: null, pauseResume: null, stop: true },
    stopped: { label: "Stopped", pulse: "pulse-stopped", generate: "Generate", pauseResume: null, stop: false },
    done: { label: "Done", pulse: "pulse-done", generate: "Generate again", pauseResume: null, stop: false },
    error: { label: "Error", pulse: "pulse-stopped", generate: "Retry", pauseResume: null, stop: false },
  };

  function initDashboard() {
    const stream = document.getElementById("manuscript-stream");
    if (!stream) return;

    const el = (id) => document.getElementById(id);
    const wordCounter = el("word-counter");
    const wordFill = el("word-progress-fill");
    const beatIdentifier = el("beat-identifier");
    const pulse = el("run-pulse");
    const runLabel = el("run-status-label");
    const runNotice = el("run-notice");
    const generateButton = el("generate-button");
    const pauseResumeButton = el("pause-resume-button");
    const stopButton = el("stop-button");
    const criticSummary = el("critic-summary");
    const criticBody = el("critic-body");
    const criticReasoning = el("critic-reasoning");
    const criticTools = el("critic-tools");
    const reviewBanner = el("review-banner");
    const reviewText = el("review-text");
    const reviewMeta = el("review-meta");
    const doneCard = el("done-card");

    let currentBeat = null;
    let currentStatus = "idle";
    let autoScroll = true;
    const beatText = new Map();
    const MAX_TOOL_CARDS = 8;

    stream.addEventListener("scroll", () => {
      const distance = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
      autoScroll = distance < 80;
    });

    function scrollIfNeeded() {
      if (autoScroll) stream.scrollTop = stream.scrollHeight;
    }

    function notice(message) {
      if (!runNotice) return;
      runNotice.textContent = message || "";
      runNotice.hidden = !message;
    }

    function setPhase(phase) {
      document.querySelectorAll("[data-phase-step]").forEach((step) => {
        step.classList.toggle("active", step.dataset.phaseStep === phase);
      });
    }

    function applyRunState(status) {
      const ui = RUN_UI[status] || RUN_UI.idle;
      currentStatus = status in RUN_UI ? status : "idle";
      if (runLabel) runLabel.textContent = ui.label;
      if (pulse) pulse.className = `pulse-dot ${ui.pulse}`;

      if (generateButton) {
        generateButton.hidden = ui.generate === null;
        if (ui.generate !== null) {
          generateButton.textContent = ui.generate;
          generateButton.disabled = false;
        }
      }
      if (pauseResumeButton) {
        pauseResumeButton.hidden = ui.pauseResume === null;
        if (ui.pauseResume !== null) {
          pauseResumeButton.dataset.runAction = ui.pauseResume;
          pauseResumeButton.textContent = ui.pauseResume === "pause" ? "Pause" : "Resume";
          pauseResumeButton.disabled = false;
        }
      }
      if (stopButton) {
        stopButton.hidden = !ui.stop;
        stopButton.disabled = false;
      }

      if (currentStatus !== "review" && reviewBanner) reviewBanner.hidden = true;
      if (currentStatus === "running") {
        if (doneCard) doneCard.hidden = true;
      }
      if (currentStatus === "idle") setPhase(null);
    }

    function setBeatIdentifier(text) {
      if (beatIdentifier) beatIdentifier.textContent = text;
    }

    function setWordProgress(count, target) {
      if (wordCounter) wordCounter.textContent = `${fmt.format(count || 0)} / ${fmt.format(target || 0)} words`;
      if (wordFill) {
        const pct = target > 0 ? Math.min(100, (count / target) * 100) : 0;
        wordFill.style.width = `${pct}%`;
      }
    }

    function ensureBeat(beatId, label) {
      const id = beatId || "unknown-beat";
      let block = stream.querySelector(`[data-beat-id="${CSS.escape(id)}"]`);
      if (!block) {
        stream.querySelector(".empty-state")?.remove();
        block = document.createElement("div");
        block.className = "beat-block";
        block.dataset.beatId = id;
        const labelEl = document.createElement("div");
        labelEl.className = "beat-label";
        labelEl.textContent = label || id;
        const proseEl = document.createElement("div");
        proseEl.className = "beat-prose";
        block.append(labelEl, proseEl);
        stream.appendChild(block);
      } else if (label) {
        block.querySelector(".beat-label").textContent = label;
      }
      currentBeat = id;
      return block;
    }

    function showReview(data) {
      if (!reviewBanner) return;
      reviewBanner.hidden = false;
      if (reviewText) reviewText.value = data.best_seen_draft || "";
      if (reviewMeta) {
        const failures = Array.isArray(data.failures) ? data.failures.length : 0;
        const best = data.best_seen_failure_count;
        reviewMeta.textContent =
          `The draft failed its quality gate ${failures ? `with ${failures} open issue(s)` : ""}` +
          `${best != null ? ` — best attempt had ${best} failure(s)` : ""}. ` +
          "Edit it if needed, then accept or regenerate.";
      }
      reviewBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const handlers = {
      hydration(snapshot) {
        const entries = Object.entries(snapshot || {});
        // Apply run_status last: it decides which panels are visible.
        entries.sort(([a], [b]) => (a === "run_status") - (b === "run_status"));
        entries.forEach(([type, data]) => {
          if (type !== "hydration") dispatch(type, data);
        });
      },
      run_status(data) {
        applyRunState(data.status || "idle");
        if (data.status === "error" && data.error) notice(`Run failed: ${data.error}`);
      },
      phase_change(data) {
        setPhase(data.phase);
      },
      chapters_planned(data) {
        setBeatIdentifier(`${data.arc_id || "Arc"} · Ch ${data.active_chapter_id || "—"} · Beat —`);
      },
      beats_planned(data) {
        const beats = Array.isArray(data.beats) ? data.beats : [];
        const active = beats.find((beat) => beat.id === data.active_beat_id) || beats[0];
        if (active) setBeatIdentifier(`${data.chapter_id || "Chapter"} · Beat ${active.ordering}`);
      },
      beat_start(data) {
        ensureBeat(data.beat_id, `Beat ${data.ordering ?? "—"} · ${data.beat_id || "unknown"}`);
        beatText.set(data.beat_id, "");
        scrollIfNeeded();
      },
      token(data) {
        const beatId = data.beat_id || currentBeat || "unknown-beat";
        const block = ensureBeat(beatId);
        const next = (beatText.get(beatId) || "") + (data.text || "");
        beatText.set(beatId, next);
        renderMarkdownInto(block.querySelector(".beat-prose"), next);
        scrollIfNeeded();
      },
      revision(data) {
        const block = ensureBeat(data.beat_id, `Revised · ${data.beat_id || "unknown"}`);
        beatText.set(data.beat_id, data.text || "");
        renderMarkdownInto(block.querySelector(".beat-prose"), data.text || "");
      },
      audit(data) {
        const failures = Array.isArray(data.failures) ? data.failures.length : 0;
        if (criticSummary) {
          criticSummary.textContent = failures
            ? `Programmatic audit found ${failures} issue(s).`
            : "Programmatic audit clean.";
        }
      },
      critic_tool(data) {
        if (!criticTools) return;
        if (criticBody) criticBody.hidden = false;
        const card = document.createElement("div");
        card.className = "critic-tool";
        const name = document.createElement("strong");
        name.textContent = data.tool || "web_search";
        const query = document.createElement("div");
        query.className = "critic-tool-query";
        query.textContent = (data.arguments && data.arguments.query) || "No query";
        const results = Array.isArray(data.result) ? data.result : [];
        const preview = document.createElement("pre");
        preview.textContent = results.length
          ? results.slice(0, 2).map((item) => `${item.title || "Result"}: ${item.snippet || item.url || ""}`).join("\n")
          : "No results returned.";
        card.append(name, query, preview);
        criticTools.prepend(card);
        while (criticTools.children.length > MAX_TOOL_CARDS) criticTools.lastChild.remove();
      },
      critic_reasoning(data) {
        if (criticBody) criticBody.hidden = false;
        if (criticReasoning) criticReasoning.textContent = data.text || "Critic returned no text.";
      },
      critic_summary(data) {
        if (criticSummary) criticSummary.textContent = data.summary || "Critic complete.";
      },
      word_count(data) {
        setWordProgress(data.word_count || 0, data.target || 0);
      },
      pointer_update(data) {
        const pointer = data.fsm_pointer || {};
        setBeatIdentifier(`Arc ${pointer.arc_id || "—"} · Ch ${pointer.chapter_id || "—"} · Beat ${pointer.beat_index ?? "—"}`);
      },
      review_needed(data) {
        showReview(data || {});
      },
      manuscript_ready(data) {
        applyRunState("done");
        if (doneCard) {
          doneCard.hidden = false;
          const words = document.getElementById("done-words");
          const path = document.getElementById("done-path");
          if (words) words.textContent = `${fmt.format(data.word_count || 0)} words`;
          if (path) path.textContent = data.path || "—";
        }
      },
    };

    function dispatch(type, data) {
      const handler = handlers[type];
      if (!handler) return;
      try {
        handler(data || {});
      } catch (err) {
        console.error(`MuseAI: handler for '${type}' failed`, err);
      }
    }

    const events = new EventSource("/stream");
    Object.keys(handlers).forEach((type) => {
      events.addEventListener(type, (event) => {
        let data = {};
        try {
          data = JSON.parse(event.data || "{}");
        } catch {
          return;
        }
        dispatch(type, data);
      });
    });
    events.onerror = () => notice("Live connection lost — reconnecting…");
    events.onopen = () => notice("");

    async function runAction(button, action) {
      button.disabled = true;
      const body = await postJSON(action === "generate" ? "/generate" : `/control/${action}`);
      if (body.aborted) return;
      if (!body.ok) {
        notice(body.error || "Request failed.");
        applyRunState(body.status || currentStatus);
        return;
      }
      notice(action === "pause" && body.status === "running" ? "Pausing at the next safe boundary…" : "");
      applyRunState(body.status || "idle");
    }

    document.querySelectorAll("[data-run-action]").forEach((button) => {
      button.addEventListener("click", () => runAction(button, button.dataset.runAction));
    });

    document.getElementById("review-accept")?.addEventListener("click", () => submitReview("accept"));
    document.getElementById("review-regenerate")?.addEventListener("click", () => submitReview("regenerate"));

    async function submitReview(decision) {
      const result = document.getElementById("review-result");
      const body = await postJSON("/control/review", { decision, edited_text: reviewText?.value || "" });
      if (body.aborted) return;
      setInlineResult(result, body.ok ? "" : body.error, body.ok);
      if (body.ok) applyRunState(body.status || "running");
    }

    // The manager is the source of truth at page load; the SSE snapshot may
    // predate a server restart.
    getJSON("/status").then((body) => {
      if (!body || !body.ok) return;
      applyRunState(body.status || "idle");
      if (typeof body.project_word_total === "number") {
        setWordProgress(body.project_word_total, body.word_target || 0);
      }
      const pointer = body.pointer;
      if (pointer && (pointer.arc_id || pointer.chapter_id)) {
        setBeatIdentifier(`Arc ${pointer.arc_id || "—"} · Ch ${pointer.chapter_id || "—"} · Beat ${pointer.beat_index ?? "—"}`);
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /* Seed                                                                */
  /* ------------------------------------------------------------------ */

  function validateSeed(raw) {
    if (!raw.trim()) return { ok: false, message: "Paste seed JSON to validate it.", neutral: true };
    let seed;
    try {
      seed = JSON.parse(raw);
    } catch (err) {
      return { ok: false, message: `Malformed JSON: ${err.message}` };
    }
    if (typeof seed !== "object" || seed === null || Array.isArray(seed)) {
      return { ok: false, message: "Seed must be a JSON object." };
    }
    const project = seed.project;
    if (typeof project !== "object" || project === null || typeof project.id !== "string" || !project.id) {
      return { ok: false, message: "seed.project.id (a string) is required." };
    }
    if (!Array.isArray(seed.arcs) || seed.arcs.length === 0) {
      return { ok: false, message: "seed.arcs must be a non-empty list." };
    }
    for (let i = 0; i < seed.arcs.length; i += 1) {
      const arc = seed.arcs[i];
      if (typeof arc !== "object" || arc === null || typeof arc.description !== "string") {
        return { ok: false, message: `seed.arcs[${i + 1}].description is required.` };
      }
    }
    for (const collection of ["threads", "characters"]) {
      if (collection in seed && !Array.isArray(seed[collection])) {
        return { ok: false, message: `seed.${collection} must be a list.` };
      }
    }
    const characters = Array.isArray(seed.characters) ? seed.characters.length : 0;
    const threads = Array.isArray(seed.threads) ? seed.threads.length : 0;
    return {
      ok: true,
      message: `Valid seed — project “${project.id}”, ${seed.arcs.length} arc(s), ${characters} character(s), ${threads} thread(s).`,
    };
  }

  function initSeed() {
    const editor = document.getElementById("seed_json");
    if (!editor) return;
    const validation = document.getElementById("seed-validation");
    const submit = document.getElementById("seed-submit");
    const example = document.getElementById("seed-example")?.dataset.example || "";
    let timer = null;

    function runValidation() {
      const verdict = validateSeed(editor.value);
      if (validation) {
        validation.textContent = verdict.message;
        validation.classList.toggle("is-ok", verdict.ok);
        validation.classList.toggle("is-error", !verdict.ok && !verdict.neutral);
      }
      if (submit) submit.disabled = !verdict.ok;
      return verdict;
    }

    editor.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(runValidation, 250);
    });

    document.getElementById("seed-format")?.addEventListener("click", () => {
      const verdict = runValidation();
      if (!verdict.ok) return;
      editor.value = JSON.stringify(JSON.parse(editor.value), null, 2);
      runValidation();
    });

    document.getElementById("seed-reset")?.addEventListener("click", () => {
      if (example) editor.value = example;
      runValidation();
    });

    runValidation();
  }

  /* ------------------------------------------------------------------ */
  /* Settings                                                            */
  /* ------------------------------------------------------------------ */

  function initSettings() {
    const page = document.getElementById("settings-page");
    if (!page) return;
    let baseConfig = {};
    try {
      baseConfig = JSON.parse(page.dataset.config || "{}");
    } catch {
      baseConfig = {};
    }
    const result = document.getElementById("settings-save-result");
    const endpointResult = document.getElementById("endpoint-test-result");

    function readGenerationValues() {
      const values = {};
      let firstError = null;
      document.querySelectorAll(".generation-setting").forEach((input) => {
        const key = input.dataset.generationKey;
        const raw = input.value.trim();
        const value = raw === "" ? NaN : Number(raw);
        const min = Number(input.min);
        const max = Number(input.max);
        const bad = !Number.isFinite(value) || value < min || value > max;
        input.classList.toggle("is-invalid", bad);
        if (bad && !firstError) {
          const label = document.querySelector(`label[for="${input.id}"]`);
          firstError = `${label ? label.textContent : key} must be a number between ${input.min} and ${input.max}.`;
        }
        values[key] = value;
      });
      return { values, error: firstError };
    }

    document.getElementById("save-settings")?.addEventListener("click", async () => {
      const { values, error } = readGenerationValues();
      if (error) {
        setInlineResult(result, error, false);
        return;
      }
      const cfg = structuredClone(baseConfig);
      cfg.endpoint.base_url = document.getElementById("setting-base-url").value.trim();
      cfg.endpoint.model_name = document.getElementById("setting-model-name").value.trim();
      cfg.endpoint.api_key = document.getElementById("setting-api-key").value;
      cfg.endpoint.tokenizer_family = document.getElementById("setting-tokenizer").value;
      Object.assign(cfg.generation, values);

      setInlineResult(result, "Saving…");
      const body = await postJSON("/settings/save", cfg);
      if (body.aborted) return;
      setInlineResult(result, body.ok ? "Settings saved." : body.error, body.ok);
    });

    document.getElementById("test-endpoint")?.addEventListener("click", async () => {
      setInlineResult(endpointResult, "Testing…");
      const body = await postJSON("/settings/test_endpoint");
      if (body.aborted) return;
      setInlineResult(endpointResult, body.ok ? `Connected — model responded: ${body.model}` : body.error, body.ok);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
    initSeed();
    initSettings();
  });
})();
