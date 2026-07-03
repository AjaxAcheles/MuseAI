/* Module: M17 (Web UI & Real-time Observer Surface)
   Planning-workbench client: project setup form, readable planning timeline,
   macro-outline approval, live SSE event stream with filters, and run
   controls. Plain JavaScript only — no framework, no build step. */

(function () {
  "use strict";

  var MAX_LOG_ENTRIES = 100;
  var NEAR_BOTTOM_PX = 40;

  var el = function (id) { return document.getElementById(id); };

  var state = {
    status: "idle",
    awaitingApproval: false,
    hasRunEnded: false,
    eventFilter: "all",
    selectedNodeId: null,
  };

  /* ---- status ribbon + buttons ---------------------------------------- */

  function setStatus(status) {
    if (!status) { return; }
    state.status = status;
    if (status === "completed" || status === "stopped" || status === "error") {
      state.hasRunEnded = true;
    }
    if (status !== "blocked") { state.awaitingApproval = false; }

    var pill = el("run-status");
    pill.textContent = status;
    pill.className = "pill pill-" + status;
    el("live-dot").className = "dot dot-" + status;

    updateControls();
  }

  function premiseFilled() {
    return el("inp-premise").value.trim().length > 0;
  }

  function updateControls() {
    var s = state.status;
    var activeish = s === "running" || s === "starting";
    var runOver = s === "idle" || s === "completed" || s === "stopped" || s === "error";

    var startBtn = el("btn-start");
    startBtn.disabled = !(runOver && premiseFilled());
    startBtn.textContent = state.hasRunEnded ? "Start new run" : "Start";
    startBtn.title = premiseFilled() ? "" : "Enter a premise first";

    el("btn-pause").disabled = !activeish;
    el("btn-resume").disabled = s !== "paused";
    el("btn-stop").disabled = !(activeish || s === "paused" || s === "blocked");

    // The form stays editable only between runs.
    ["inp-title", "inp-premise", "inp-genre", "inp-words",
     "sel-exec-mode", "sel-approval-mode"].forEach(function (id) {
      el(id).disabled = !runOver;
    });

    updateApprovalCard();
  }

  function updateApprovalCard() {
    var card = el("approval-card");
    var button = el("btn-approve");
    if (state.awaitingApproval) {
      card.classList.remove("hidden");
      button.classList.remove("hidden");
      button.disabled = false;
    } else {
      button.classList.add("hidden");
      if (!card.dataset.keepVisible) { card.classList.add("hidden"); }
    }
  }

  function setPointer(ptr) {
    ptr = ptr || {};
    el("ptr-arc").textContent = ptr.arc_id || "–";
    el("ptr-chapter").textContent = ptr.chapter_id || "–";
    el("ptr-scene").textContent = ptr.scene_id || "–";
    el("ptr-beat").textContent = ptr.beat_id ? ptr.beat_id + " (#" + ptr.beat_index + ")" : "–";
  }

  function setPhase(phase) {
    if (phase) { el("phase").textContent = phase; }
  }

  /* ---- helpers ----------------------------------------------------------- */

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function showBlocked(reason, note) {
    var card = el("block-card");
    var body = el("block-body");
    if (!reason || reason === "awaiting_macro_approval") {
      // The approval gate has its own card; this one is for hard blocks.
      card.classList.add("hidden");
      body.textContent = "";
      return;
    }
    body.innerHTML = escapeHtml("Reason: " + reason) + (note ? "<br>" + escapeHtml(note) : "");
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

  /* ---- planning timeline --------------------------------------------------- */

  function nodeButton(node, currentId) {
    // Runtime PlanningNode ids are "<snapshot>:<level>:<id>" while the FSM
    // pointer carries the bare id, so match on the id suffix too.
    var isCurrent = !!currentId && (
      node.node_id === currentId ||
      (typeof node.node_id === "string" && node.node_id.indexOf(":" + currentId) !== -1 &&
       node.node_id.lastIndexOf(":" + currentId) === node.node_id.length - (":" + currentId).length)
    );
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "plan-node";
    btn.dataset.nodeId = node.node_id;
    btn.innerHTML =
      '<span class="node-level">' + escapeHtml(node.level) + "</span>" +
      escapeHtml(node.title || node.node_id) +
      (node.status === "stub" ? '<span class="node-stub">stub — not planned yet</span>' : "") +
      (isCurrent ? '<span class="node-current">◀ current</span>' : "") +
      (node.summary ? '<span class="node-summary">' + escapeHtml(node.summary) + "</span>" : "");
    btn.addEventListener("click", function () { selectNode(node.node_id, btn); });
    return btn;
  }

  function renderTreeInto(container, node, currentId) {
    var li = document.createElement("li");
    li.appendChild(nodeButton(node, currentId));
    if (node.children && node.children.length) {
      var ul = document.createElement("ul");
      node.children.forEach(function (child) { renderTreeInto(ul, child, currentId); });
      li.appendChild(ul);
    }
    container.appendChild(li);
  }

  function renderTimeline(data) {
    var meta = el("timeline-meta");
    var body = el("timeline-body");
    var runtime = el("timeline-runtime");

    if (!data || !data.snapshot_id) {
      meta.textContent = (data && data.message) || "No planning snapshot yet.";
      meta.classList.add("muted");
      body.textContent = "";
      runtime.textContent = "No scenes or beats planned yet.";
      return;
    }
    meta.classList.remove("muted");
    meta.textContent =
      data.snapshot_id + " · " + data.status +
      (data.approved_at ? " (approved)" : "") +
      " · " + data.mode + " · " + data.revision_count + " revisions";

    body.textContent = "";
    if (data.tree) {
      var tree = document.createElement("ul");
      tree.className = "plan-tree";
      renderTreeInto(tree, data.tree, null);
      body.appendChild(tree);
    } else {
      body.textContent = "No global plan yet.";
    }

    runtime.textContent = "";
    var scenes = (data.runtime && data.runtime.scenes) || [];
    var beats = (data.runtime && data.runtime.beats) || [];
    if (!scenes.length && !beats.length) {
      runtime.classList.add("muted");
      runtime.textContent = "No scenes or beats planned yet.";
    } else {
      runtime.classList.remove("muted");
      var list = document.createElement("ul");
      list.className = "plan-tree";
      scenes.forEach(function (scene) {
        renderTreeInto(list, scene, data.runtime.current_scene && "" + data.runtime.current_scene);
      });
      beats.forEach(function (beat) {
        renderTreeInto(list, beat, data.runtime.current_beat && "" + data.runtime.current_beat);
      });
      runtime.appendChild(list);
    }

    // Re-mark the selected node if it is still present.
    if (state.selectedNodeId) {
      var again = document.querySelector('[data-node-id="' + state.selectedNodeId + '"]');
      if (again) { again.classList.add("node-selected"); }
    }
  }

  var timelineFetchQueued = false;
  function refreshTimeline() {
    if (timelineFetchQueued) { return; }
    timelineFetchQueued = true;
    setTimeout(function () {
      timelineFetchQueued = false;
      fetch("/plan/snapshot")
        .then(function (resp) { return resp.json(); })
        .then(renderTimeline)
        .catch(function () { /* transient; next snapshot event retries */ });
    }, 200);
  }

  function selectNode(nodeId, btn) {
    state.selectedNodeId = nodeId;
    document.querySelectorAll(".plan-node.node-selected").forEach(function (other) {
      other.classList.remove("node-selected");
    });
    if (btn) { btn.classList.add("node-selected"); }

    fetch("/plan/node/" + encodeURIComponent(nodeId))
      .then(function (resp) { return resp.json(); })
      .then(function (payload) {
        if (!payload.ok) { notice(payload.message || "Node not found."); return; }
        var node = payload.node;
        el("node-detail-card").classList.remove("hidden");
        el("node-detail-title").textContent = node.level + " · " + (node.title || node.node_id);
        var rows = (node.fields || []).map(function (field) {
          var value = field.value;
          if (typeof value === "object") { value = JSON.stringify(value, null, 2); }
          return "<tr><th>" + escapeHtml(field.name) + '</th><td class="field-value">' +
            escapeHtml(String(value)) + "</td></tr>";
        });
        el("node-detail-body").innerHTML = rows.length
          ? "<table>" + rows.join("") + "</table>"
          : '<span class="muted">No readable fields for this node yet.</span>';
        el("node-detail-raw").textContent = JSON.stringify(node.raw || {}, null, 2);
      })
      .catch(function (err) { notice("Node detail failed: " + err); });
  }

  /* ---- event log --------------------------------------------------------- */

  var FILTER_MAP = {
    status: ["status", "phase_change", "done"],
    planning: ["planning_node", "planning_snapshot", "pointer_update"],
    approval: ["approval_state"],
    error: ["error"],
  };

  function eventCategory(type, data) {
    if (type === "planning_blocked") {
      return data && data.awaiting_approval ? "approval" : "error";
    }
    var categories = Object.keys(FILTER_MAP);
    for (var i = 0; i < categories.length; i += 1) {
      if (FILTER_MAP[categories[i]].indexOf(type) !== -1) { return categories[i]; }
    }
    return "status";
  }

  function applyEventFilter() {
    var rows = el("event-log").children;
    for (var i = 0; i < rows.length; i += 1) {
      var row = rows[i];
      row.style.display =
        state.eventFilter === "all" || row.dataset.category === state.eventFilter
          ? "" : "none";
    }
  }

  function logEvent(type, data, timestamp) {
    var log = el("event-log");
    var nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < NEAR_BOTTOM_PX;

    var line = document.createElement("div");
    line.className = "event-line";
    line.dataset.category = eventCategory(type, data);

    var time = timestamp ? new Date(timestamp).toLocaleTimeString() : "";
    var message = (data && data.message) || "";
    var details = "";
    try { details = JSON.stringify(data, null, 2); } catch (e) { details = String(data); }

    line.innerHTML =
      '<div class="event-head">' +
      '<span class="event-time">' + escapeHtml(time) + "</span>" +
      '<span class="event-type event-type-' + escapeHtml(type) + '">' + escapeHtml(type) + "</span>" +
      '<span class="event-message">' + escapeHtml(message) + "</span>" +
      "</div>" +
      "<details><summary>details</summary><pre class=\"event-json\">" +
      escapeHtml(details) + "</pre></details>";

    if (state.eventFilter !== "all" && line.dataset.category !== state.eventFilter) {
      line.style.display = "none";
    }
    log.appendChild(line);
    while (log.children.length > MAX_LOG_ENTRIES) {
      log.removeChild(log.firstChild);
    }
    if (nearBottom) { log.scrollTop = log.scrollHeight; }
  }

  el("event-filters").addEventListener("click", function (ev) {
    var target = ev.target.closest(".filter-btn");
    if (!target) { return; }
    state.eventFilter = target.dataset.filter;
    document.querySelectorAll(".filter-btn").forEach(function (btn) {
      btn.classList.toggle("filter-active", btn === target);
    });
    applyEventFilter();
  });

  /* ---- snapshot application ---------------------------------------------- */

  function applySnapshot(snap) {
    if (!snap) { return; }
    setPhase(snap.phase);
    setPointer(snap.pointer);
    showBlocked(snap.planning_block_reason);
    showError(snap.last_error);
    if (snap.awaiting_planning_approval) { showApprovalAwaiting(null); }
    if (snap.status) { setStatus(snap.status); }
  }

  function applyManager(mgr) {
    if (!mgr) { return; }
    if (mgr.awaiting_approval) { showApprovalAwaiting(null); }
    setStatus(mgr.status);
    if (mgr.last_error) { showError(mgr.last_error); }
  }

  function showApprovalAwaiting(message) {
    state.awaitingApproval = true;
    var card = el("approval-card");
    delete card.dataset.keepVisible;
    card.classList.remove("hidden");
    el("approval-body").textContent =
      message ||
      "The macro outline (global plan, arcs, chapters) is ready. Review the planning " +
      "timeline below, then approve to continue into scene and beat planning.";
    updateControls();
  }

  function showApprovalResolved(message) {
    state.awaitingApproval = false;
    var card = el("approval-card");
    card.dataset.keepVisible = "1";
    el("approval-body").textContent = message || "Macro outline approved.";
    updateControls();
  }

  /* ---- SSE ---------------------------------------------------------------- */

  var handlers = {
    status: function (d) { setStatus(d.status); },
    phase_change: function (d) { setPhase(d.phase); },
    pointer_update: function (d) { setPointer(d); },
    planning_snapshot: function (d) { refreshTimeline(); },
    planning_node: function (d) { /* readable row already in the event log */ },
    planning_blocked: function (d) {
      if (d.awaiting_approval) { showApprovalAwaiting(d.message); }
      else { showBlocked(d.reason); }
    },
    approval_state: function (d) {
      if (d.awaiting_approval) { showApprovalAwaiting(d.message); }
      else { showApprovalResolved(d.message); }
    },
    done: function (d) { notice(d.message || "Run complete."); refreshTimeline(); },
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
          applySnapshot(parsed.data);   // connection snapshot: re-render everything
          refreshTimeline();
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
    };
  }

  /* ---- controls ------------------------------------------------------------ */

  function post(path, payload) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    })
      .then(function (resp) { return resp.json(); })
      .then(function (body) {
        notice(body.ok ? null : body.message || "Request refused.");
        if (body.status) { setStatus(body.status); }
        return body;
      })
      .catch(function (err) { notice("Request failed: " + err); });
  }

  function startPayload() {
    var payload = {
      premise: el("inp-premise").value.trim(),
      planning_execution_mode: el("sel-exec-mode").value,
      approval_mode: el("sel-approval-mode").value,
    };
    var title = el("inp-title").value.trim();
    var genre = el("inp-genre").value.trim();
    var words = el("inp-words").value;
    if (title) { payload.title = title; }
    if (genre) { payload.genre = genre; }
    if (words) { payload.target_word_count = parseInt(words, 10); }
    return payload;
  }

  el("btn-start").addEventListener("click", function () {
    showError(null);
    showBlocked(null);
    el("approval-card").classList.add("hidden");
    delete el("approval-card").dataset.keepVisible;
    state.selectedNodeId = null;
    post("/control/start", startPayload());
  });
  el("btn-pause").addEventListener("click", function () { post("/control/pause"); });
  el("btn-resume").addEventListener("click", function () { post("/control/resume"); });
  el("btn-stop").addEventListener("click", function () { post("/control/stop"); });
  el("btn-approve").addEventListener("click", function () {
    el("btn-approve").disabled = true;
    post("/plan/approve").then(function (body) {
      if (body && !body.ok) { el("btn-approve").disabled = false; }
    });
  });

  el("inp-premise").addEventListener("input", updateControls);

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

  refreshTimeline();
  connectEvents();
  updateControls();
})();
