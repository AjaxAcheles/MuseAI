/* Module: M17 (Web UI & Real-time Observer Surface)
   Dashboard client: fetch /api/status on load, keep the page live over SSE,
   and drive the start/pause/resume/stop controls. Plain JavaScript only. */

(function () {
  "use strict";

  var MAX_LOG_ENTRIES = 100;
  var NEAR_BOTTOM_PX = 40;

  var el = function (id) { return document.getElementById(id); };

  var state = {
    status: "idle",
    running: false,
    awaitingApproval: false,
  };

  /* ---- status ribbon + buttons ---------------------------------------- */

  function setStatus(status, running) {
    if (status) { state.status = status; }
    if (typeof running === "boolean") { state.running = running; }

    var pill = el("run-status");
    pill.textContent = state.status;
    pill.className = "pill pill-" + state.status;

    var dot = el("live-dot");
    dot.className = "dot dot-" + state.status;

    updateButtons();
  }

  function updateButtons() {
    var s = state.status;
    var activeish = s === "running" || s === "starting";
    el("btn-start").disabled = activeish || s === "paused";
    el("btn-pause").disabled = !activeish;
    el("btn-resume").disabled = s !== "paused";
    el("btn-stop").disabled = !(activeish || s === "paused" || s === "blocked");
  }

  function setPointer(ptr) {
    ptr = ptr || {};
    el("ptr-arc").textContent = ptr.arc_id || "–";
    el("ptr-chapter").textContent = ptr.chapter_id || "–";
    el("ptr-scene").textContent = ptr.scene_id || "–";
    var beat = ptr.beat_id ? ptr.beat_id + " (#" + ptr.beat_index + ")" : "–";
    el("ptr-beat").textContent = beat;
  }

  function setPhase(phase) {
    if (phase) { el("phase").textContent = phase; }
  }

  /* ---- cards ------------------------------------------------------------ */

  function renderKeyValueTable(obj) {
    var rows = Object.keys(obj)
      .filter(function (k) { return obj[k] !== null && obj[k] !== undefined; })
      .map(function (k) {
        var v = obj[k];
        if (typeof v === "object") { v = JSON.stringify(v); }
        return "<tr><th>" + escapeHtml(k) + "</th><td>" + escapeHtml(String(v)) + "</td></tr>";
      });
    return "<table>" + rows.join("") + "</table>";
  }

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderPlanningSnapshot(data) {
    if (!data) { return; }
    el("planning-snapshot-body").classList.remove("muted");
    el("planning-snapshot-body").innerHTML = renderKeyValueTable(data);
  }

  function renderPlanningNode(data) {
    if (!data) { return; }
    el("planning-node-body").classList.remove("muted");
    el("planning-node-body").innerHTML = renderKeyValueTable(data);
  }

  function showBlocked(reason, note) {
    var card = el("block-card");
    var body = el("block-body");
    if (!reason) {
      card.classList.add("hidden");
      body.textContent = "";
      return;
    }
    var text = "Reason: " + reason;
    if (reason === "awaiting_macro_approval") {
      note = note || "Approval gate reached. Approval resume is not implemented in this vertical slice.";
    }
    body.innerHTML = escapeHtml(text) + (note ? "<br>" + escapeHtml(note) : "");
    card.classList.remove("hidden");
  }

  function showError(message) {
    var card = el("error-card");
    var body = el("error-body");
    if (!message) {
      card.classList.add("hidden");
      body.textContent = "";
      return;
    }
    body.textContent = message;
    card.classList.remove("hidden");
  }

  function notice(message) {
    var box = el("notice");
    if (!message) { box.classList.add("hidden"); box.textContent = ""; return; }
    box.textContent = message;
    box.classList.remove("hidden");
  }

  /* ---- event log --------------------------------------------------------- */

  function logEvent(type, data, timestamp) {
    var log = el("event-log");
    var nearBottom =
      log.scrollHeight - log.scrollTop - log.clientHeight < NEAR_BOTTOM_PX;

    var line = document.createElement("div");
    line.className = "event-line";
    var time = timestamp ? new Date(timestamp).toLocaleTimeString() : "";
    var detail = "";
    try { detail = JSON.stringify(data); } catch (e) { detail = String(data); }
    if (detail && detail.length > 240) { detail = detail.slice(0, 240) + "…"; }
    line.innerHTML =
      '<span class="event-time">' + escapeHtml(time) + "</span>" +
      '<span class="event-type event-type-' + escapeHtml(type) + '">' + escapeHtml(type) + "</span>" +
      '<span class="event-detail">' + escapeHtml(detail) + "</span>";
    log.appendChild(line);

    while (log.children.length > MAX_LOG_ENTRIES) {
      log.removeChild(log.firstChild);
    }
    // Only follow the tail if the user is already near the bottom.
    if (nearBottom) { log.scrollTop = log.scrollHeight; }
  }

  /* ---- snapshot application ---------------------------------------------- */

  function applySnapshot(snap) {
    if (!snap) { return; }
    setPhase(snap.phase);
    setPointer(snap.pointer);
    renderPlanningSnapshot(snap.planning_snapshot);
    renderPlanningNode(snap.planning_node);
    showBlocked(snap.planning_block_reason);
    showError(snap.last_error);
  }

  function applyManager(mgr) {
    if (!mgr) { return; }
    setStatus(mgr.status, mgr.running);
    if (mgr.last_error) { showError(mgr.last_error); }
  }

  /* ---- SSE ---------------------------------------------------------------- */

  var handlers = {
    status: function (d) {
      setStatus(d.status, d.running);
      if (d.status === "stopped" || d.status === "completed") {
        // A finished run is no longer blocked; keep errors visible.
        if (state.status !== "blocked") { showBlocked(null); }
      }
    },
    phase_change: function (d) { setPhase(d.phase); },
    pointer_update: function (d) { setPointer(d); },
    planning_snapshot: function (d) { renderPlanningSnapshot(d); },
    planning_node: function (d) { renderPlanningNode(d); },
    planning_blocked: function (d) { showBlocked(d.reason); },
    approval_state: function (d) {
      state.awaitingApproval = !!d.awaiting_approval;
      if (d.awaiting_approval) { showBlocked("awaiting_macro_approval", d.note); }
    },
    done: function (d) { notice(d.message || "Run complete."); },
    error: function (d) { showError(d.message || JSON.stringify(d)); },
  };

  function connectEvents() {
    var es = new EventSource("/events");
    Object.keys(handlers).forEach(function (type) {
      es.addEventListener(type, function (ev) {
        var parsed;
        try { parsed = JSON.parse(ev.data); } catch (e) { return; }
        logEvent(type, parsed.data, parsed.timestamp);
        if (type === "status" && parsed.data && parsed.data.pointer !== undefined) {
          // Connection snapshot event: re-render everything.
          applySnapshot(parsed.data);
          if (parsed.data.status) { setStatus(parsed.data.status, parsed.data.running); }
          return;
        }
        handlers[type](parsed.data || {});
      });
    });
    es.onopen = function () {
      var conn = el("conn-status");
      conn.textContent = "live";
      conn.className = "conn conn-live";
    };
    es.onerror = function () {
      var conn = el("conn-status");
      conn.textContent = "Reconnecting…";
      conn.className = "conn conn-reconnecting";
      // EventSource retries automatically; nothing else to do here.
    };
  }

  /* ---- controls ------------------------------------------------------------ */

  function post(path) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    })
      .then(function (resp) { return resp.json(); })
      .then(function (body) {
        if (!body.ok) {
          notice(body.message || "Request refused.");
        } else {
          notice(null);
        }
        if (body.status) { setStatus(body.status); }
        return body;
      })
      .catch(function (err) { notice("Request failed: " + err); });
  }

  el("btn-start").addEventListener("click", function () {
    showError(null);
    showBlocked(null);
    post("/control/start");
  });
  el("btn-pause").addEventListener("click", function () { post("/control/pause"); });
  el("btn-resume").addEventListener("click", function () { post("/control/resume"); });
  el("btn-stop").addEventListener("click", function () { post("/control/stop"); });

  /* ---- boot ------------------------------------------------------------------ */

  var bootstrapNode = el("bootstrap-data");
  if (bootstrapNode) {
    try {
      var boot = JSON.parse(bootstrapNode.textContent);
      applySnapshot(boot.snapshot);
      applyManager(boot.manager);
    } catch (e) { /* server-rendered bootstrap is best-effort */ }
  }

  fetch("/api/status")
    .then(function (resp) { return resp.json(); })
    .then(function (body) {
      if (!body || !body.ok) { return; }
      applySnapshot(body.snapshot);
      applyManager(body.manager);
    })
    .catch(function () { /* SSE connection below still reports liveness */ });

  connectEvents();
  updateButtons();
})();
