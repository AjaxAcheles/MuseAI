/**
 * Page controllers for MuseAI.
 *
 * The load-bearing rule of this file: the Committed Story panel is written *only* from
 * `GET /committed`. Stream tokens, revisions, critic notes, and best-seen review drafts are
 * confined to the Live Activity panel, because none of them are manuscript until the commit
 * node says so.
 *
 * `/status` is authoritative. Anything the stream implies about run state is reconciled against
 * it on load and after a reconnect.
 */
"use strict";

(function (MuseAI) {
  const { getJSON, postJSON, renderMarkdownInto, escapeHtml, formatNumber, emptyState, toast, setInlineResult } =
    MuseAI;

  const byId = (id) => document.getElementById(id);
  const MAX_ACTIVITY_ENTRIES = 50;

  /** How each manager status presents in the command bar. Mirrors `RunStatus` in manager.py. */
  const RUN_UI = {
    idle: { label: "Idle", pulse: "pulse-idle", generate: true, pauseResume: null, stop: false },
    running: { label: "Running", pulse: "pulse-running", generate: false, pauseResume: "pause", stop: true },
    paused: { label: "Paused", pulse: "pulse-paused", generate: false, pauseResume: "resume", stop: true },
    review: { label: "Review needed", pulse: "pulse-review", generate: false, pauseResume: null, stop: true },
    stopped: { label: "Stopped", pulse: "pulse-stopped", generate: true, pauseResume: null, stop: false },
    done: { label: "Done", pulse: "pulse-done", generate: true, pauseResume: null, stop: false },
    error: { label: "Error", pulse: "pulse-error", generate: true, pauseResume: null, stop: false },
  };

  /** Which FSM node is speaking, for the role badge. */
  const NODE_ROLE = {
    plan_chapter: "planner",
    plan_beat: "planner",
    assemble_context: "system",
    draft_prose: "drafter",
    audit: "critic",
    critics: "critic",
    revise: "reviser",
    commit: "committer",
  };

  /** Activity filter buckets. Every server event maps to exactly one. */
  const EVENT_CATEGORY = {
    phase_change: "planner",
    chapters_planned: "planner",
    beats_planned: "planner",
    pad_update: "planner",
    beat_start: "drafter",
    token: "drafter",
    revision: "drafter",
    audit: "critics",
    critic_tool: "critics",
    critic_reasoning: "critics",
    critic_summary: "critics",
    critic_health: "warnings",
    planner_repaired: "warnings",
    word_count: "commit",
    pointer_update: "commit",
    manuscript_ready: "commit",
    run_status: "commit",
    review_needed: "warnings",
  };

  /* ========================================================= seed workspace */

  /**
   * Wire one copy of the seed workspace partial. `ns` matches the macro's namespace, so the
   * Dashboard drawer and the Seed & Plan page share every behaviour below.
   */
  function initSeedWorkspace(ns, options) {
    const config = options || {};
    const editor = byId(`${ns}-json`);
    if (!editor) return null;

    const seedApi = MuseAI.seed;
    const premise = byId(`${ns}-premise`);
    const timeline = byId(`${ns}-timeline`);
    const validation = byId(`${ns}-validation`);
    const errorList = byId(`${ns}-errors`);
    const exampleSource = document.querySelector("[data-example]");
    const exampleSeed = exampleSource ? exampleSource.dataset.example : "";

    let lastValid = null;

    function showErrors(errors) {
      if (!errorList) return;
      errorList.innerHTML = errors.map((message) => `<li>${escapeHtml(message)}</li>`).join("");
    }

    /** Validate the editor's contents; returns the parsed seed or null. Drives every button. */
    function refresh() {
      const parsed = seedApi.parseSeedJson(editor.value);
      if (!parsed.ok) {
        lastValid = null;
        setInlineResult(validation, parsed.error, false);
        showErrors([]);
        seedApi.renderSeedTimeline(timeline, null);
        return null;
      }

      const check = seedApi.validateSeedClientSide(parsed.value);
      seedApi.renderSeedTimeline(timeline, parsed.value);
      if (!check.ok) {
        lastValid = null;
        setInlineResult(validation, `${check.errors.length} problem${check.errors.length === 1 ? "" : "s"} found.`, false);
        showErrors(check.errors);
        return null;
      }

      lastValid = parsed.value;
      const summary = seedApi.summarizeSeed(parsed.value);
      setInlineResult(
        validation,
        `Valid seed · ${summary.arcs} arc${summary.arcs === 1 ? "" : "s"} · ` +
          `${summary.threads} thread${summary.threads === 1 ? "" : "s"} · ` +
          `${summary.characters} character${summary.characters === 1 ? "" : "s"}`,
        true
      );
      showErrors([]);
      return parsed.value;
    }

    let debounce = null;
    editor.addEventListener("input", () => {
      window.clearTimeout(debounce);
      debounce = window.setTimeout(refresh, 200);
    });

    const format = byId(`${ns}-format`);
    if (format) {
      format.addEventListener("click", () => {
        const formatted = seedApi.formatSeedJson(editor.value);
        if (formatted.ok) editor.value = formatted.value;
        refresh();
      });
    }

    const validate = byId(`${ns}-validate`);
    if (validate) validate.addEventListener("click", refresh);

    const reset = byId(`${ns}-reset`);
    if (reset) {
      reset.addEventListener("click", () => {
        editor.value = exampleSeed;
        refresh();
      });
    }

    const applyPremise = byId(`${ns}-apply-premise`);
    if (applyPremise && premise) {
      applyPremise.addEventListener("click", () => {
        const text = premise.value.trim();
        if (!text) {
          setInlineResult(validation, "Write a premise before applying it.", false);
          return;
        }
        const parsed = seedApi.parseSeedJson(editor.value);
        const base = parsed.ok ? parsed.value : {};
        editor.value = JSON.stringify(seedApi.applyPremise(base, text), null, 2);
        refresh();
        toast("Premise applied. Complete and validate the JSON seed before loading.", "info");
      });
    }

    // The timeline pane is hidden at first paint, so its SVG would size against a zero-width box.
    // Re-render when the pane actually becomes visible.
    const timelinePane = byId(`${ns}-pane-timeline`);
    if (timelinePane) timelinePane.addEventListener("tab:shown", () => seedApi.renderSeedTimeline(timeline, lastValid));

    /** Submit to the backend. The server validates again, and its error replaces ours. */
    async function submit(resultElement) {
      const seed = refresh();
      if (!seed) {
        setInlineResult(resultElement, "Fix the seed before loading.", false);
        return null;
      }
      setInlineResult(resultElement, "Loading seed…", true);
      const response = await seedApi.submitSeedToBackend(seed);
      if (!response.ok) {
        setInlineResult(resultElement, response.error, false);
        return null;
      }
      const summary = seedApi.summarizeSeed(seed);
      setInlineResult(resultElement, "Seed loaded.", true);
      toast(
        `Seed loaded: ${summary.title} · ${summary.arcs} arc${summary.arcs === 1 ? "" : "s"} · ` +
          `${summary.threads} thread${summary.threads === 1 ? "" : "s"} · ` +
          `${summary.characters} character${summary.characters === 1 ? "" : "s"}`,
        "ok"
      );
      if (config.onLoaded) config.onLoaded(response, seed);
      return response;
    }

    refresh();
    return { refresh, submit, getSeed: () => lastValid };
  }

  /* ============================================================== seed page */

  function initSeedPage() {
    const page = byId("seed-page");
    if (!page) return;

    const workspace = initSeedWorkspace("page", { onLoaded: () => renderLoadedSeed() });
    const loadButton = byId("page-load-seed");
    const result = byId("page-load-result");
    if (loadButton && workspace) loadButton.addEventListener("click", () => workspace.submit(result));

    /** Show what the backend actually holds, not what is typed in the editor. */
    async function renderLoadedSeed() {
      const target = byId("seed-summary");
      if (!target) return;
      const status = await getJSON("/status");
      if (!status.ok || !status.seed_loaded) {
        target.innerHTML = emptyState("No seed loaded", "Load a seed to see its summary here.");
        return;
      }
      const project = status.project || {};
      const counts = status.counts || {};
      target.innerHTML = `
        <dl class="summary-list">
          <dt>Project</dt><dd>${escapeHtml(project.id)}</dd>
          ${project.genre ? `<dt>Genre</dt><dd>${escapeHtml(project.genre)}</dd>` : ""}
          <dt>Arcs</dt><dd>${counts.arcs || 0}</dd>
          <dt>Threads</dt><dd>${counts.threads || 0}</dd>
          <dt>Characters</dt><dd>${counts.characters || 0}</dd>
          ${project.word_count_target ? `<dt>Word target</dt><dd>${formatNumber(project.word_count_target)}</dd>` : ""}
        </dl>
        ${project.premise ? `<p class="summary-premise">${escapeHtml(project.premise)}</p>` : ""}
        <p class="summary-hint">Committed words so far: ${formatNumber(status.project_word_total || 0)}.</p>`;
    }

    renderLoadedSeed();
  }

  /* ============================================================== dashboard */

  function initDashboard() {
    const generateButton = byId("generate-button");
    if (!generateButton) return;

    const pauseResumeButton = byId("pause-resume-button");
    const stopButton = byId("stop-button");
    const runLabel = byId("run-status-label");
    const runPulse = byId("run-pulse");
    const streamDot = byId("stream-dot");
    const notice = byId("run-notice");
    const criticWarning = byId("critic-warning");
    const criticWarningDetail = byId("critic-warning-detail");
    const seedPill = byId("seed-pill");

    const committedStory = byId("committed-story");
    const committedWords = byId("committed-words");
    const committedLast = byId("committed-last");

    const liveActivity = byId("live-activity");
    const liveStream = byId("live-stream");
    const streamBox = byId("stream-box");
    const activityRole = byId("activity-role");
    const activityPhase = byId("activity-phase");

    const reviewBanner = byId("review-banner");
    const reviewText = byId("review-text");
    const reviewMeta = byId("review-meta");
    const reviewResult = byId("review-result");
    const doneCard = byId("done-card");

    const rail = byId("progress-rail");
    const railCaption = byId("rail-caption");

    let activityFilter = "all";
    let entries = [];
    let seedLoaded = false;
    let runStatus = "idle";
    let issueCount = 0;
    let streamBuffer = "";
    let currentTarget = null;

    /* ----------------------------------------------------------- command bar */

    function showNotice(message, isError) {
      if (!notice) return;
      notice.hidden = !message;
      notice.textContent = message || "";
      notice.classList.toggle("is-error", Boolean(isError));
    }

    // Latched so the toast fires on the transition into degraded, not on every beat.
    let criticDegraded = false;

    /**
     * Reflect the backend's `critic_health`. The banner is a pure function of the
     * latest event, so a recovered critic clears it without a special "recovered"
     * message. The wording must not soften what degraded means: those beats
     * committed with only the programmatic audit behind them.
     */
    function applyCriticHealth(data) {
      if (!criticWarning) return;
      const degraded = Boolean(data.degraded);

      if (degraded && !criticDegraded) {
        toast(`Continuity critic degraded after ${data.streak} unreadable replies.`, "warning");
      }
      criticDegraded = degraded;
      criticWarning.hidden = !degraded;
      if (!degraded) return;

      const parsed = data.lenient_used
        ? "Its latest reply was salvaged with relaxed validation."
        : "Its latest reply could not be read at all.";
      criticWarningDetail.textContent =
        `The critic has returned unreadable output for ${data.streak} beats in a row ` +
        `(threshold ${data.threshold}). ${parsed} ` +
        `Beats are still committing, but continuity is no longer being checked — ` +
        `only the automated passive-voice audit is. Check the model in Settings.`;
    }

    function applyRunState(status) {
      runStatus = status;
      const ui = RUN_UI[status] || RUN_UI.idle;
      if (runLabel) runLabel.textContent = ui.label;
      if (runPulse) runPulse.className = `pulse-dot ${ui.pulse}`;

      // Generate stays disabled until the backend confirms a seed exists.
      generateButton.disabled = !(ui.generate && seedLoaded);
      generateButton.textContent = status === "running" ? "Running…" : "Generate";
      generateButton.title = seedLoaded ? "" : "Load a seed before generating.";

      if (pauseResumeButton) {
        pauseResumeButton.hidden = ui.pauseResume === null;
        if (ui.pauseResume) {
          pauseResumeButton.dataset.runAction = ui.pauseResume;
          pauseResumeButton.textContent = ui.pauseResume === "pause" ? "Pause" : "Resume";
        }
      }
      if (stopButton) stopButton.hidden = !ui.stop;
      if (reviewBanner && status !== "review") reviewBanner.hidden = true;

      const phaseCard = byId("card-phase-value");
      if (phaseCard && (status === "idle" || status === "done" || status === "stopped")) {
        phaseCard.textContent = ui.label;
      }
    }

    function applySeedState(status) {
      seedLoaded = Boolean(status.seed_loaded);
      const project = status.project || {};
      if (seedPill) {
        seedPill.textContent = seedLoaded ? `Seed loaded: ${project.id}` : "No seed loaded";
        seedPill.classList.toggle("is-missing", !seedLoaded);
        seedPill.classList.toggle("is-loaded", seedLoaded);
      }
      const seedState = byId("card-seed-state");
      const seedMeta = byId("card-seed-meta");
      if (seedState) seedState.textContent = seedLoaded ? "Loaded" : "Missing";
      if (seedMeta) {
        seedMeta.textContent = seedLoaded ? [project.id, project.genre].filter(Boolean).join(" · ") : "Load a seed to begin.";
      }
    }

    function applyProgress(total, target) {
      const words = total || 0;
      const value = byId("card-progress-value");
      const bar = byId("card-progress-bar");
      const fill = byId("card-progress-fill");
      const meta = byId("card-progress-meta");
      if (value) value.textContent = `${formatNumber(words)} words`;
      if (meta) meta.textContent = target ? `Target ${formatNumber(target)} words.` : "No target set.";
      if (bar && fill) {
        const percent = target ? Math.min(100, Math.round((words / target) * 100)) : 0;
        fill.style.width = `${percent}%`;
        bar.setAttribute("aria-valuenow", String(words));
        bar.setAttribute("aria-valuemax", String(target || 0));
      }
      if (committedWords) committedWords.textContent = `${formatNumber(words)} words`;
    }

    function applyLastCommit(lastCommit) {
      const value = byId("card-commit-value");
      const meta = byId("card-commit-meta");
      if (!lastCommit) {
        if (value) value.textContent = "None";
        if (meta) meta.textContent = "Nothing has been committed.";
        if (committedLast) committedLast.textContent = "no commits yet";
        return;
      }
      if (value) value.textContent = lastCommit.beat_id;
      if (meta) meta.textContent = `${formatNumber(lastCommit.word_count)} words · ${lastCommit.chapter_id}`;
      if (committedLast) committedLast.textContent = `last commit ${lastCommit.beat_id}`;
    }

    function applyEndpoint(endpoint) {
      const state = byId("card-endpoint-state");
      const meta = byId("card-endpoint-meta");
      const dot = byId("card-endpoint-dot");
      if (!endpoint) return;
      const configured = Boolean(endpoint.base_url && endpoint.model_name);
      if (state) state.textContent = configured ? "Configured" : "Missing";
      if (dot) dot.className = `health-dot ${configured ? "is-configured" : "is-missing"}`;
      if (meta) meta.textContent = configured ? `${endpoint.model_name} · ${endpoint.base_url}` : "Set a base URL and model.";
    }

    /* ------------------------------------------------------ committed story */

    /** Refetch committed prose. This is the only writer of the Committed Story panel. */
    async function refreshCommitted() {
      const payload = await getJSON("/committed");
      if (!payload.ok || !committedStory) return;

      const beats = payload.beats || [];
      applyProgress(payload.project_word_total, currentTarget);

      if (beats.length === 0) {
        committedStory.innerHTML = emptyState(
          "No committed story yet",
          "Generation will appear here after a beat passes review and commits."
        );
        return;
      }

      // Group by chapter so the manuscript reads as chapters, not a flat list of beats.
      const chapters = [];
      beats.forEach((beat) => {
        const last = chapters[chapters.length - 1];
        if (last && last.id === beat.chapter_id) last.beats.push(beat);
        else
          chapters.push({
            id: beat.chapter_id,
            ordering: beat.chapter_ordering,
            description: beat.chapter_description,
            beats: [beat],
          });
      });

      committedStory.innerHTML = chapters
        .map(
          (chapter) => `
          <section class="beat-block">
            <h3 class="beat-label">Chapter ${chapter.ordering}</h3>
            ${chapter.description ? `<p class="chapter-description">${escapeHtml(chapter.description)}</p>` : ""}
            ${chapter.beats.map((beat) => `<div class="beat-prose" data-beat="${escapeHtml(beat.beat_id)}"></div>`).join("")}
          </section>`
        )
        .join("");

      beats.forEach((beat) => {
        const node = committedStory.querySelector(`[data-beat="${CSS.escape(beat.beat_id)}"]`);
        renderMarkdownInto(node, beat.prose);
      });
    }

    /* ----------------------------------------------------------------- rail */

    /**
     * Draw the story rail from `/outline`. Arcs are large nodes; chapters the planners have
     * created are small ones. Nothing is drawn that the database does not contain, so before
     * planning runs a seeded project shows arcs only.
     */
    async function refreshRail() {
      if (!rail) return;
      const outline = await getJSON("/outline");
      if (!outline.ok) return;

      const arcs = outline.arcs || [];
      if (arcs.length === 0) {
        rail.innerHTML = emptyState("No story shape yet", "Load a seed to see its arcs.");
        if (railCaption) railCaption.textContent = "Load a seed to see the story shape.";
        return;
      }

      const pointer = outline.pointer || {};
      const nodes = [];
      arcs.forEach((arc, arcIndex) => {
        const arcWords = arc.chapters.reduce(
          (sum, chapter) =>
            sum + chapter.beats.filter((beat) => beat.status === "completed").reduce((n, beat) => n + beat.word_count, 0),
          0
        );
        nodes.push({
          kind: "arc",
          status: arc.status,
          label: `Arc ${arcIndex + 1}`,
          detail: arc.description,
          words: arcWords,
          active: pointer.arc_id === arc.id && !pointer.chapter_id,
        });
        arc.chapters.forEach((chapter, chapterIndex) => {
          const done = chapter.beats.filter((beat) => beat.status === "completed");
          nodes.push({
            kind: "chapter",
            status: chapter.status,
            label: `Ch ${chapterIndex + 1}`,
            detail: chapter.description,
            words: done.reduce((n, beat) => n + beat.word_count, 0),
            beats: chapter.beats.length,
            committedBeats: done.length,
            active: pointer.chapter_id === chapter.id,
          });
        });
      });

      const activeIndex = nodes.findIndex((node) => node.active);
      const gap = 110;
      const left = 40;
      const width = left + Math.max(nodes.length, 1) * gap;
      const y = 54;

      const markup = nodes
        .map((node, index) => {
          const cx = left + index * gap;
          const radius = node.kind === "arc" ? 15 : 9;
          const classes = ["rail-node", `is-${node.status}`];
          if (activeIndex >= 0 && index < activeIndex) classes.push("is-passed");
          if (node.active) classes.push(runStatus === "review" ? "is-blocked" : "is-active");

          const tip = [
            node.detail || node.label,
            `Status: ${node.status}`,
            node.words ? `${formatNumber(node.words)} committed words` : "no committed words",
            node.kind === "chapter" ? `${node.committedBeats}/${node.beats} beats committed` : null,
          ]
            .filter(Boolean)
            .join("\n");

          return `
            <g class="${classes.join(" ")}" tabindex="0" role="listitem"
               aria-label="${escapeHtml(node.label)}, status ${escapeHtml(node.status)}">
              <title>${escapeHtml(tip)}</title>
              <circle class="rail-${node.kind}" cx="${cx}" cy="${y}" r="${radius}"></circle>
              <text class="rail-label" x="${cx}" y="${y + 34}" text-anchor="middle">${escapeHtml(node.label)}</text>
            </g>`;
        })
        .join("");

      const lastX = left + Math.max(nodes.length - 1, 0) * gap;
      const fillTo = activeIndex >= 0 ? left + activeIndex * gap : left;
      rail.innerHTML = `
        <svg class="rail" viewBox="0 0 ${width} 100" role="list" aria-label="Story progress rail">
          <line class="rail-track" x1="${left}" y1="${y}" x2="${lastX}" y2="${y}"></line>
          <line class="rail-fill" x1="${left}" y1="${y}" x2="${fillTo}" y2="${y}"></line>
          ${markup}
        </svg>`;

      if (railCaption) {
        const active = nodes[activeIndex];
        railCaption.textContent = active
          ? `Currently at ${active.label}${pointer.beat_index !== undefined ? ` · beat ${pointer.beat_index + 1}` : ""}`
          : `${arcs.length} arc${arcs.length === 1 ? "" : "s"} planned.`;
      }
    }

    /* -------------------------------------------------------- live activity */

    function renderActivity() {
      if (!liveActivity) return;
      const visible = entries.filter((entry) => activityFilter === "all" || entry.category === activityFilter);
      if (visible.length === 0) {
        liveActivity.innerHTML = emptyState("No activity yet", "Events appear here while the engine runs.");
        return;
      }
      liveActivity.innerHTML = visible
        .map(
          (entry) => `
          <div class="activity-entry is-${entry.category}">
            <span class="activity-time">${entry.time}</span>
            <span class="activity-role-tag">${escapeHtml(entry.role)}</span>
            <span class="activity-text">${escapeHtml(entry.text)}</span>
          </div>`
        )
        .join("");
      liveActivity.scrollTop = 0;
    }

    /** Record one line of backend activity. Newest first, capped so the DOM stays small. */
    function logActivity(type, text, role) {
      entries.unshift({
        category: EVENT_CATEGORY[type] || "commit",
        role: role || "system",
        text,
        time: new Date().toLocaleTimeString(),
      });
      if (entries.length > MAX_ACTIVITY_ENTRIES) entries.length = MAX_ACTIVITY_ENTRIES;
      renderActivity();
    }

    function setRole(role, phase) {
      if (activityRole && role) activityRole.textContent = role;
      if (activityPhase && phase) activityPhase.textContent = phase;
      const badge = byId("card-role-badge");
      const phaseValue = byId("card-phase-value");
      if (badge && role) badge.textContent = role;
      if (phaseValue && phase) phaseValue.textContent = phase;
    }

    function bumpIssues(count) {
      issueCount += count;
      const value = byId("card-issues-value");
      const meta = byId("card-issues-meta");
      if (value) value.textContent = String(issueCount);
      if (meta) {
        meta.textContent =
          runStatus === "review"
            ? "Review needed."
            : issueCount === 0
            ? "No audit failures seen."
            : "Audit failures seen this session.";
      }
    }

    function showReview(data) {
      if (!reviewBanner) return;
      reviewBanner.hidden = false;
      if (reviewText) reviewText.value = data.best_seen_draft || "";
      if (reviewMeta) {
        const failures = (data.failures || []).length;
        reviewMeta.textContent = `${failures} unresolved issue${failures === 1 ? "" : "s"}. Edit the draft, then accept or regenerate.`;
      }
    }

    /* ------------------------------------------------------------------ SSE */

    const handlers = {
      hydration(snapshot) {
        // Replay the last payload of each type; run_status last so it wins the command bar.
        Object.keys(snapshot)
          .sort((a, b) => (a === "run_status" ? 1 : b === "run_status" ? -1 : 0))
          .forEach((type) => dispatch(type, snapshot[type]));
      },

      run_status(data) {
        applyRunState(data.status);
        if (data.error) {
          // The failed run's draft is salvaged to disk; tell the user where.
          const salvaged = data.draft_path ? ` Draft saved to ${data.draft_path}` : "";
          showNotice(`${data.error}${salvaged}`, true);
          logActivity("run_status", `Error: ${data.error}`, "system");
        }
        if (data.status === "done" || data.status === "stopped") refreshCommitted();
      },

      phase_change(data) {
        const role = NODE_ROLE[data.node] || "system";
        setRole(role, data.phase);
        logActivity("phase_change", String(data.phase), role);
      },

      chapters_planned(data) {
        logActivity("chapters_planned", `Planned ${data.chapter_count} chapters for ${data.arc_id}`, "planner");
        refreshRail();
      },

      beats_planned(data) {
        logActivity("beats_planned", `Planned ${data.beat_count} beats for ${data.chapter_id}`, "planner");
        refreshRail();
      },

      pad_update(data) {
        const pad = data.target_pad || {};
        const axes = ["pleasure", "arousal", "dominance"]
          .filter((axis) => pad[axis] !== undefined)
          .map((axis) => `${axis[0].toUpperCase()} ${Number(pad[axis]).toFixed(2)}`)
          .join(" · ");
        logActivity("pad_update", `Target emotion for ${data.character_id}: ${axes || "unset"}`, "planner");
      },

      beat_start(data) {
        streamBuffer = "";
        if (streamBox) streamBox.hidden = false;
        if (liveStream) liveStream.textContent = "";
        logActivity("beat_start", `Drafting beat ${data.beat_id} (target ${data.word_target} words)`, "drafter");
      },

      // Raw model output. Belongs here and nowhere else.
      token(data) {
        if (!liveStream) return;
        streamBuffer += data.text || "";
        liveStream.textContent = streamBuffer;
        liveStream.scrollTop = liveStream.scrollHeight;
      },

      revision(data) {
        streamBuffer = data.text || "";
        if (liveStream) liveStream.textContent = streamBuffer;
        logActivity("revision", `Revised beat ${data.beat_id} (${data.mode}, retry ${data.retry_count})`, "reviser");
      },

      audit(data) {
        const failures = (data.failures || []).length;
        bumpIssues(failures);
        logActivity(
          "audit",
          failures === 0
            ? `Audit passed (passive density ${Number(data.passive_density).toFixed(2)})`
            : `Audit found ${failures} issue${failures === 1 ? "" : "s"}`,
          "critic"
        );
      },

      critic_tool(data) {
        const query = (data.arguments && data.arguments.query) || "";
        logActivity("critic_tool", `Searched: ${query}`, "critic");
      },

      critic_reasoning(data) {
        logActivity("critic_reasoning", data.text || "", "critic");
      },

      critic_summary(data) {
        logActivity("critic_summary", data.summary || `${data.total_failures} failures`, "critic");
      },

      critic_health(data) {
        applyCriticHealth(data);
        if (data.streak > 0) {
          const detail = data.degraded ? " Continuity checking is degraded." : "";
          logActivity(
            "critic_health",
            `Critic output unreadable (${data.streak} in a row).${detail}`,
            "critic"
          );
        }
      },

      /**
       * The planner's JSON was unreadable and we rewrote its quoting to salvage
       * it. The plan that resulted is not literally what the model returned, so
       * say so rather than letting it pass as a normal plan.
       */
      planner_repaired(data) {
        const what = data.what || "plan";
        logActivity(
          "planner_repaired",
          `Repaired malformed JSON from the ${data.agent || "planner"} to recover ` +
            `${data.count} ${what}. The model did not return valid JSON.`,
          "planner"
        );
        toast(`Recovered a malformed ${what} plan by repairing its JSON.`, "warning");
      },

      word_count(data) {
        currentTarget = data.target || currentTarget;
        applyProgress(data.word_count, currentTarget);
        refreshCommitted();
      },

      pointer_update(data) {
        logActivity("pointer_update", `Committed beat ${data.beat_id}`, "committer");
        refreshCommitted();
        refreshRail();
        refreshStatus();
      },

      review_needed(data) {
        applyRunState("review");
        showReview(data);
        bumpIssues(0);
        logActivity("review_needed", "Parked for human review", "system");
      },

      manuscript_ready(data) {
        if (doneCard) doneCard.hidden = false;
        const words = byId("done-words");
        const path = byId("done-path");
        if (words) words.textContent = `${formatNumber(data.word_count)} words`;
        if (path) path.textContent = data.path;
        logActivity("manuscript_ready", `Manuscript exported to ${data.path}`, "system");
        refreshCommitted();
      },
    };

    function dispatch(type, data) {
      const handler = handlers[type];
      if (!handler) return;
      try {
        handler(data);
      } catch (error) {
        // A malformed payload must not tear down the stream.
        console.error(`handler for ${type} failed`, error);
      }
    }

    function connectStream() {
      const events = new EventSource("/stream");
      Object.keys(handlers).forEach((type) => {
        events.addEventListener(type, (event) => {
          let data = {};
          try {
            data = JSON.parse(event.data);
          } catch (error) {
            return;
          }
          dispatch(type, data);
        });
      });
      events.onopen = () => {
        if (streamDot) streamDot.className = "stream-dot is-connected";
        showNotice("");
      };
      events.onerror = () => {
        if (streamDot) streamDot.className = "stream-dot is-disconnected";
        showNotice("Event stream disconnected. Retrying…", true);
        // EventSource reconnects on its own; re-read authoritative state when it does.
        window.setTimeout(refreshStatus, 2000);
      };
    }

    /* -------------------------------------------------------------- controls */

    async function runAction(action) {
      const url = action === "generate" ? "/generate" : `/control/${action}`;
      const response = await postJSON(url, {});
      if (!response.ok) {
        showNotice(response.error, true);
        return;
      }
      showNotice("");
      if (response.status) applyRunState(response.status);
    }

    document.querySelectorAll("[data-run-action]").forEach((button) => {
      button.addEventListener("click", () => runAction(button.dataset.runAction));
    });

    async function submitReview(decision) {
      const payload = { decision };
      if (decision === "accept" && reviewText) payload.edited_text = reviewText.value;
      setInlineResult(reviewResult, "Submitting…", true);
      const response = await postJSON("/control/review", payload);
      if (!response.ok) {
        setInlineResult(reviewResult, response.error, false);
        return;
      }
      setInlineResult(reviewResult, "", true);
      if (reviewBanner) reviewBanner.hidden = true;
      applyRunState(response.status);
    }

    const accept = byId("review-accept");
    const regenerate = byId("review-regenerate");
    if (accept) accept.addEventListener("click", () => submitReview("accept"));
    if (regenerate) regenerate.addEventListener("click", () => submitReview("regenerate"));

    document.querySelectorAll("[data-activity-filter]").forEach((chip) => {
      chip.addEventListener("click", () => {
        activityFilter = chip.dataset.activityFilter;
        document.querySelectorAll("[data-activity-filter]").forEach((other) => other.classList.toggle("active", other === chip));
        renderActivity();
      });
    });

    /* ----------------------------------------------------------- seed drawer */

    const drawer = MuseAI.createDrawer(byId("seed-drawer"));
    const drawerResult = byId("drawer-load-result");
    const drawerWorkspace = initSeedWorkspace("drawer", {
      onLoaded: async () => {
        if (drawer) drawer.close();
        await refreshStatus();
        await refreshRail();
        await refreshCommitted();
      },
    });

    const openDrawer = byId("open-seed-drawer");
    if (openDrawer && drawer) openDrawer.addEventListener("click", drawer.open);

    const drawerValidate = byId("drawer-validate-seed");
    if (drawerValidate && drawerWorkspace) {
      drawerValidate.addEventListener("click", () => {
        const seed = drawerWorkspace.refresh();
        setInlineResult(drawerResult, seed ? "Seed is valid." : "Seed is not valid yet.", Boolean(seed));
      });
    }

    const drawerLoad = byId("drawer-load-seed");
    if (drawerLoad && drawerWorkspace) drawerLoad.addEventListener("click", () => drawerWorkspace.submit(drawerResult));

    /* ------------------------------------------------------------ hydration */

    /** `/status` is the source of truth. Everything else is a hint. */
    async function refreshStatus() {
      const status = await getJSON("/status");
      if (!status.ok) {
        showNotice(status.error || "Could not read run status.", true);
        return;
      }
      currentTarget = status.word_target;
      applySeedState(status);
      applyRunState(status.status);
      applyProgress(status.project_word_total, status.word_target);
      applyLastCommit(status.last_commit);
      applyEndpoint(status.endpoint);
      bumpIssues(0);
    }

    renderActivity();
    refreshStatus().then(() => {
      refreshCommitted();
      refreshRail();
      connectStream();
    });
  }

  /* =============================================================== settings */

  function initSettings() {
    const page = byId("settings-page");
    if (!page) return;

    const baseConfig = JSON.parse(page.dataset.config || "{}");
    const saveButton = byId("save-settings");
    const saveResult = byId("settings-save-result");

    // Save stays disabled until something actually changes.
    document.querySelectorAll(".dirty-watch").forEach((input) => {
      const event = input.type === "checkbox" || input.tagName === "SELECT" ? "change" : "input";
      input.addEventListener(event, () => {
        if (saveButton) saveButton.disabled = false;
        setInlineResult(saveResult, "", true);
      });
    });

    /** Read number inputs, naming the offending field rather than silently clamping it. */
    function readNumbers(selector, datasetKey) {
      const values = {};
      for (const input of document.querySelectorAll(selector)) {
        const value = Number(input.value);
        const min = Number(input.min);
        const max = Number(input.max);
        if (Number.isNaN(value) || value < min || value > max) {
          const label = document.querySelector(`label[for="${input.id}"]`);
          const name = label ? label.textContent.trim() : input.id;
          throw new Error(`${name} must be between ${min} and ${max}.`);
        }
        values[input.dataset[datasetKey]] = value;
      }
      return values;
    }

    if (saveButton) {
      saveButton.addEventListener("click", async () => {
        let generation;
        let endpointNumbers;
        try {
          generation = readNumbers(".generation-setting", "generationKey");
          endpointNumbers = readNumbers(".endpoint-setting", "endpointKey");
        } catch (error) {
          setInlineResult(saveResult, error.message, false);
          return;
        }

        const payload = structuredClone(baseConfig);
        payload.endpoint = {
          ...payload.endpoint,
          ...endpointNumbers,
          base_url: byId("setting-base-url").value.trim(),
          model_name: byId("setting-model-name").value.trim(),
          tokenizer_family: byId("setting-tokenizer").value,
          // An empty string means "keep the key already configured"; the server substitutes it.
          api_key: byId("setting-api-key").value,
        };
        payload.generation = { ...payload.generation, ...generation };
        payload.log_level = byId("setting-log_level").value;
        payload.web_search_timeout = Number(byId("setting-web_search_timeout").value);
        payload.allow_reset = byId("setting-allow_reset").checked;

        setInlineResult(saveResult, "Saving…", true);
        const response = await postJSON("/settings/save", payload);
        if (!response.ok) {
          setInlineResult(saveResult, response.error, false);
          return;
        }
        setInlineResult(saveResult, "Settings saved.", true);
        saveButton.disabled = true;
        toast("Settings saved.", "ok");
      });
    }

    const testButton = byId("test-endpoint");
    if (testButton) {
      testButton.addEventListener("click", async () => {
        const result = byId("endpoint-test-result");
        const dot = byId("endpoint-health-dot");
        setInlineResult(result, "Testing…", true);
        if (dot) dot.className = "health-dot is-testing";
        const response = await postJSON("/settings/test_endpoint", {});
        setInlineResult(result, response.ok ? `Reached ${response.model}.` : response.error, response.ok);
        if (dot) dot.className = `health-dot ${response.ok ? "is-ok" : "is-failed"}`;
      });
    }

    const resetButton = byId("reset-project");
    if (resetButton) {
      resetButton.addEventListener("click", async () => {
        if (!window.confirm("Reset deletes the project database and event log. Continue?")) return;
        const response = await postJSON("/control/reset", {});
        setInlineResult(byId("reset-result"), response.ok ? "Project data reset." : response.error, response.ok);
      });
    }
  }

  /* =============================================================== database */

  function initDatabase() {
    const page = byId("database-page");
    if (!page) return;

    const typeSelect = byId("db-type");
    const search = byId("db-search");
    const head = byId("db-head");
    const body = byId("db-body");
    const count = byId("db-count");
    const drawer = MuseAI.createDrawer(byId("record-drawer"));
    const recordJson = byId("record-json");
    let records = [];

    const preview = (value) => {
      if (value === null || value === undefined) return "";
      const text = String(value);
      return text.length > 80 ? `${text.slice(0, 80)}…` : text;
    };

    function render(columns) {
      if (records.length === 0) {
        head.innerHTML = "";
        body.innerHTML = `<tr><td>${emptyState("No records", "Nothing of this type has been written yet.")}</td></tr>`;
        return;
      }
      head.innerHTML = `<tr>${columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join("")}</tr>`;
      body.innerHTML = records
        .map(
          (record, index) =>
            `<tr data-index="${index}" tabindex="0">${columns
              .map((column) => `<td>${escapeHtml(preview(record[column]))}</td>`)
              .join("")}</tr>`
        )
        .join("");

      body.querySelectorAll("tr[data-index]").forEach((row) => {
        const open = () => {
          if (recordJson) recordJson.textContent = JSON.stringify(records[Number(row.dataset.index)], null, 2);
          if (drawer) drawer.open();
        };
        row.addEventListener("click", open);
        row.addEventListener("keydown", (event) => {
          if (event.key === "Enter") open();
        });
      });
    }

    async function load() {
      const type = typeSelect.value;

      if (type === "__events__") {
        const payload = await getJSON("/database/event-log?limit=200");
        if (!payload.ok) {
          count.textContent = payload.error;
          return;
        }
        records = (payload.events || []).slice().reverse();
        count.textContent = `${payload.total} event${payload.total === 1 ? "" : "s"} in the log · showing ${records.length}`;
        render(["type", "beat_id", "word_count"]);
        return;
      }

      const url = `/database/records?type=${encodeURIComponent(type)}&q=${encodeURIComponent(search.value.trim())}`;
      const payload = await getJSON(url);
      if (!payload.ok) {
        count.textContent = payload.error;
        records = [];
        render([]);
        return;
      }
      records = payload.records || [];
      count.textContent = `${payload.total} record${payload.total === 1 ? "" : "s"} · showing ${records.length}`;
      render(records.length ? Object.keys(records[0]) : []);
    }

    typeSelect.addEventListener("change", load);
    byId("db-refresh").addEventListener("click", load);

    let debounce = null;
    search.addEventListener("input", () => {
      window.clearTimeout(debounce);
      debounce = window.setTimeout(load, 250);
    });

    load();
  }

  /* =================================================================== logs */

  function initLogs() {
    const page = byId("logs-page");
    if (!page) return;

    const view = byId("log-view");
    const source = byId("log-source");
    const eventsBox = byId("log-events");
    const seen = [];

    async function loadTail() {
      const payload = await getJSON(`/logs/tail?source=${encodeURIComponent(source.value)}&limit=400`);
      if (!payload.ok) {
        view.textContent = payload.error;
        return;
      }
      view.textContent = payload.lines.length
        ? payload.lines.join("\n")
        : payload.exists
        ? "The log file is empty."
        : "No log file has been written yet.";
      view.scrollTop = view.scrollHeight;
    }

    source.addEventListener("change", loadTail);
    byId("log-refresh").addEventListener("click", loadTail);
    loadTail();

    // The browser's own record of the stream, so a dropped event is visible as a gap.
    eventsBox.innerHTML = emptyState("No stream events yet", "Events this tab receives will be listed here.");
    const events = new EventSource("/stream");
    Object.keys(EVENT_CATEGORY)
      .concat(["hydration"])
      .forEach((type) => {
        events.addEventListener(type, (event) => {
          seen.unshift({ type, time: new Date().toLocaleTimeString(), data: event.data });
          if (seen.length > 100) seen.length = 100;
          eventsBox.innerHTML = seen
            .map(
              (entry) => `
              <details class="log-event">
                <summary><span class="activity-time">${entry.time}</span> <code>${escapeHtml(entry.type)}</code></summary>
                <pre>${escapeHtml(entry.data)}</pre>
              </details>`
            )
            .join("");
        });
      });
  }

  /* ================================================================ exports */

  function initExports() {
    const page = byId("exports-page");
    if (!page) return;

    const run = byId("export-run");
    const result = byId("export-result");
    if (!run) return;

    run.addEventListener("click", async () => {
      setInlineResult(result, "Exporting…", true);
      const response = await postJSON("/exports/manuscript", {});
      if (!response.ok) {
        setInlineResult(result, response.error, false);
        return;
      }
      setInlineResult(result, `Exported ${formatNumber(response.word_count)} words to ${response.path}.`, true);
      toast("Manuscript exported.", "ok");
      const download = byId("export-download");
      if (download) {
        download.classList.remove("disabled");
        download.removeAttribute("aria-disabled");
        download.removeAttribute("tabindex");
      }
    });
  }

  /* =================================================================== boot */

  document.addEventListener("DOMContentLoaded", () => {
    MuseAI.initTabs(document);
    initSeedPage();
    initDashboard();
    initSettings();
    initDatabase();
    initLogs();
    initExports();
  });
})(window.MuseAI);
