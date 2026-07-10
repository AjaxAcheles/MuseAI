/**
 * The View Chat page: one bubble per LLM call, for every agent in the engine.
 *
 * Data arrives on two paths that must agree:
 *   - `GET /chat/history` replays the on-disk transcript (chat_start/chat_end
 *     records) so a reload shows the whole conversation;
 *   - the live `/stream` SSE connection delivers chat_start, per-token
 *     chat_token deltas, and chat_end for calls made while the tab is open.
 *
 * The stream is connected *before* history is fetched and its events are held
 * in a backlog until history has rendered, so a call that completes in that
 * window is never lost — and never duplicated, because bubbles are keyed by
 * call id.
 */
"use strict";

(function (MuseAI) {
  const { getJSON, escapeHtml, renderMarkdownInto, formatNumber, emptyState } = MuseAI;

  const byId = (id) => document.getElementById(id);

  const AGENT_LABEL = {
    chapter_planner: "Chapter planner",
    beat_planner: "Beat planner",
    drafter: "Drafter",
    critic: "Critic",
    reviser: "Reviser",
    endpoint_test: "Endpoint test",
    system: "System",
  };

  // Beyond this many token <span>s, the oldest are merged into one text node.
  // The fade-in only matters for tokens still arriving; keeping thousands of
  // spans alive would make a long drafter turn crawl.
  const MAX_TOKEN_SPANS = 400;

  const CHARS_PER_TOKEN = 4; // same heuristic the tokenizer's fallback uses

  function agentLabel(agent) {
    return AGENT_LABEL[agent] || agent || "System";
  }

  function timeLabel(ts) {
    if (!ts) return "";
    return new Date(ts * 1000).toLocaleTimeString();
  }

  function promptSummary(messages) {
    const chars = messages.reduce((sum, message) => {
      const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content || "");
      return sum + content.length;
    }, 0);
    const tokens = Math.ceil(chars / CHARS_PER_TOKEN);
    return `Prompt · ${messages.length} message${messages.length === 1 ? "" : "s"} · ~${formatNumber(tokens)} tokens`;
  }

  function promptMessagesHtml(messages) {
    return messages
      .map((message) => {
        const content =
          typeof message.content === "string" ? message.content : JSON.stringify(message.content || "", null, 2);
        const toolCalls = message.tool_calls
          ? `<pre class="chat-prompt-tools">${escapeHtml(JSON.stringify(message.tool_calls, null, 2))}</pre>`
          : "";
        return `
          <div class="chat-prompt-msg">
            <span class="chat-role chat-role-${escapeHtml(message.role || "user")}">${escapeHtml(message.role || "user")}</span>
            <pre class="chat-prompt-body">${escapeHtml(content)}</pre>
            ${toolCalls}
          </div>`;
      })
      .join("");
  }

  function initChat() {
    const root = byId("chat-root");
    if (!root) return;

    const scroller = byId("chat-scroll");
    const messagesEl = byId("chat-messages");
    const connEl = byId("chat-conn");
    const filterBar = byId("chat-filters");

    /** call id -> {el, agent, thinkingDetails, thinkingBody, responseEl, footEl, done} */
    const calls = new Map();
    let activeFilter = "all";
    let emptyEl = null;

    /* ------------------------------------------------------------ scrolling */

    function pinned() {
      return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
    }

    function keepPinned(wasPinned) {
      if (wasPinned) scroller.scrollTop = scroller.scrollHeight;
    }

    /* -------------------------------------------------------------- bubbles */

    function showEmptyState() {
      if (calls.size > 0 || emptyEl) return;
      emptyEl = document.createElement("div");
      emptyEl.innerHTML = emptyState(
        "No LLM traffic yet",
        "Every prompt the engine sends — planner, drafter, critic, reviser — will appear here as it happens."
      );
      messagesEl.appendChild(emptyEl);
    }

    function clearEmptyState() {
      if (emptyEl) {
        emptyEl.remove();
        emptyEl = null;
      }
    }

    function applyFilterTo(el, agent) {
      el.classList.toggle("chat-hidden", activeFilter !== "all" && agent !== activeFilter);
    }

    /**
     * Create the bubble for one call. `start` may be null when the page joined
     * mid-call and only tokens are arriving; the prompt block is then omitted
     * because inventing one would be lying about what was sent.
     */
    function createCall(id, agent, start) {
      clearEmptyState();
      const el = document.createElement("article");
      el.className = "chat-call";
      el.dataset.agent = agent;
      applyFilterTo(el, agent);

      const promptHtml = start
        ? `<details class="chat-prompt">
             <summary>${escapeHtml(promptSummary(start.messages || []))}</summary>
             <div class="chat-prompt-messages">${promptMessagesHtml(start.messages || [])}</div>
           </details>`
        : `<p class="chat-joined-note">Joined mid-call — the prompt was sent before this page connected.</p>`;

      el.innerHTML = `
        <header class="chat-call-head">
          <span class="chat-agent chat-agent-${escapeHtml(agent)}">${escapeHtml(agentLabel(agent))}</span>
          ${start && start.model ? `<span class="chat-model">${escapeHtml(start.model)}</span>` : ""}
          ${start && start.ts ? `<span class="chat-time">${escapeHtml(timeLabel(start.ts))}</span>` : ""}
          <span class="chat-state" data-state="streaming">streaming</span>
        </header>
        ${promptHtml}
        <details class="chat-thinking" open hidden>
          <summary>Thinking</summary>
          <div class="chat-thinking-body"></div>
        </details>
        <div class="chat-response"></div>
        <footer class="chat-call-foot" hidden></footer>`;

      const entry = {
        el,
        agent,
        stateEl: el.querySelector(".chat-state"),
        thinkingDetails: el.querySelector(".chat-thinking"),
        thinkingBody: el.querySelector(".chat-thinking-body"),
        responseEl: el.querySelector(".chat-response"),
        footEl: el.querySelector(".chat-call-foot"),
        done: false,
        seenSeq: 0, // tokens up to this server sequence arrived via the history partial
      };
      calls.set(id, entry);
      messagesEl.appendChild(el);
      return entry;
    }

    /** Append one streamed token as a fading span, consolidating old spans. */
    function appendToken(container, text) {
      if (container.childNodes.length >= MAX_TOKEN_SPANS) {
        const keep = MAX_TOKEN_SPANS / 2;
        let merged = "";
        while (container.childNodes.length > keep) {
          merged += container.firstChild.textContent;
          container.firstChild.remove();
        }
        container.insertBefore(document.createTextNode(merged), container.firstChild);
      }
      const span = document.createElement("span");
      span.className = "chat-token";
      span.textContent = text;
      container.appendChild(span);
    }

    /* --------------------------------------------------------- event intake */

    function onChatStart(data) {
      if (calls.has(data.id)) return; // history already rendered this call
      const wasPinned = pinned();
      createCall(data.id, data.agent || "system", data);
      keepPinned(wasPinned);
    }

    function onChatToken(data) {
      let entry = calls.get(data.id);
      if (entry && entry.done) return;
      // The history partial already carried this token's text; appending it
      // again would duplicate everything received before history rendered.
      if (entry && data.seq && data.seq <= entry.seenSeq) return;
      const wasPinned = pinned();
      if (!entry) entry = createCall(data.id, data.agent || "system", null);
      if (entry.stateEl.dataset.state !== "streaming") {
        // History guessed "interrupted", but the call is in fact still going.
        entry.stateEl.dataset.state = "streaming";
        entry.stateEl.textContent = "streaming";
      }
      if (data.kind === "thinking") {
        entry.thinkingDetails.hidden = false;
        const body = entry.thinkingBody;
        // The thinking box scrolls on its own once it overflows; keep it
        // pinned to the newest tokens unless the reader scrolled back up.
        const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
        appendToken(body, data.text);
        if (nearBottom) body.scrollTop = body.scrollHeight;
      } else {
        appendToken(entry.responseEl, data.text);
      }
      keepPinned(wasPinned);
    }

    function finishCall(entry, end) {
      entry.done = true;
      const ok = end.ok !== false;
      entry.stateEl.dataset.state = ok ? "done" : "error";
      entry.stateEl.textContent = ok ? "done" : "error";

      if (end.thinking) {
        entry.thinkingDetails.hidden = false;
        entry.thinkingBody.textContent = end.thinking;
        const summary = entry.thinkingDetails.querySelector("summary");
        summary.textContent = `Thinking · ${formatNumber(end.thinking.length)} chars`;
      } else if (!entry.thinkingBody.hasChildNodes()) {
        entry.thinkingDetails.hidden = true;
      }
      // Deliberation collapses once the answer exists, like every chat UI.
      entry.thinkingDetails.open = false;

      if (!ok) {
        entry.responseEl.innerHTML = `<p class="chat-error">${escapeHtml(end.error || "The call failed.")}</p>`;
      } else if (end.text) {
        entry.responseEl.classList.add("rendered");
        renderMarkdownInto(entry.responseEl, end.text);
      } else if (end.tool_calls) {
        entry.responseEl.innerHTML = `<p class="chat-tool-note">Requested ${end.tool_calls} tool call${end.tool_calls === 1 ? "" : "s"} — the result feeds the next message.</p>`;
      } else if (!entry.responseEl.hasChildNodes()) {
        entry.responseEl.innerHTML = `<p class="chat-tool-note">The model returned no prose.</p>`;
      }

      const stats = [];
      if (end.tokens_in) stats.push(`${formatNumber(end.tokens_in)} tokens in`);
      if (end.tokens_out) stats.push(`${formatNumber(end.tokens_out)} tokens out`);
      if (end.finish_reason) stats.push(`finish: ${end.finish_reason}`);
      if (end.attempt > 1) stats.push(`attempt ${end.attempt}`);
      if (stats.length > 0) {
        entry.footEl.hidden = false;
        entry.footEl.textContent = stats.join(" · ");
      }
    }

    function onChatEnd(data) {
      const wasPinned = pinned();
      let entry = calls.get(data.id);
      if (!entry) entry = createCall(data.id, data.agent || "system", null);
      if (!entry.done) finishCall(entry, data);
      keepPinned(wasPinned);
    }

    /* ---------------------------------------------------- history + stream */

    function renderHistory(records) {
      const starts = new Map();
      records.forEach((record) => {
        if (record.event === "chat_start") starts.set(record.id, record);
      });

      records.forEach((record) => {
        if (record.event === "chat_start") {
          if (!calls.has(record.id)) createCall(record.id, record.agent || "system", record);
        } else if (record.event === "chat_end") {
          let entry = calls.get(record.id);
          if (!entry) entry = createCall(record.id, record.agent || "system", starts.get(record.id) || null);
          if (!entry.done) finishCall(entry, record);
        }
      });

      // A start with no end and no live stream behind it was interrupted.
      calls.forEach((entry) => {
        if (!entry.done && entry.stateEl.dataset.state === "streaming") {
          entry.stateEl.dataset.state = "interrupted";
          entry.stateEl.textContent = "interrupted";
        }
      });
    }

    /**
     * Calls still in flight when history was fetched. The transcript on disk
     * has their start but no end, so without this the thinking and prose that
     * streamed before this page loaded would simply vanish. Runs after
     * `renderHistory`, which is why it also un-marks them as interrupted.
     */
    function renderPartials(partials) {
      partials.forEach((partial) => {
        let entry = calls.get(partial.id);
        if (!entry) entry = createCall(partial.id, partial.agent || "system", partial);
        if (entry.done) return;
        entry.seenSeq = partial.seq || 0;
        if (partial.thinking) {
          entry.thinkingDetails.hidden = false;
          entry.thinkingBody.textContent = partial.thinking;
          entry.thinkingBody.scrollTop = entry.thinkingBody.scrollHeight;
        }
        if (partial.text) entry.responseEl.textContent = partial.text;
        entry.stateEl.dataset.state = "streaming";
        entry.stateEl.textContent = "streaming";
      });
    }

    let historyReady = false;
    const backlog = [];

    function intake(kind, data) {
      if (!historyReady) {
        backlog.push([kind, data]);
        return;
      }
      if (kind === "start") onChatStart(data);
      else if (kind === "token") onChatToken(data);
      else onChatEnd(data);
    }

    function connect() {
      const source = new EventSource("/stream");
      source.addEventListener("chat_start", (event) => intake("start", JSON.parse(event.data)));
      source.addEventListener("chat_token", (event) => intake("token", JSON.parse(event.data)));
      source.addEventListener("chat_end", (event) => intake("end", JSON.parse(event.data)));
      source.onopen = () => {
        connEl.dataset.state = "live";
        connEl.textContent = "Live";
      };
      source.onerror = () => {
        connEl.dataset.state = "reconnecting";
        connEl.textContent = "Reconnecting…";
      };
    }

    async function boot() {
      connect();
      try {
        const payload = await getJSON("/chat/history?limit=200");
        if (payload.ok) {
          renderHistory(payload.records || []);
          renderPartials(payload.partials || []);
        }
      } finally {
        historyReady = true;
        backlog.forEach(([kind, data]) => {
          if (kind === "start") onChatStart(data);
          else if (kind === "token") onChatToken(data);
          else onChatEnd(data);
        });
        backlog.length = 0;
        showEmptyState();
        scroller.scrollTop = scroller.scrollHeight;
      }
    }

    /* --------------------------------------------------------------- filter */

    filterBar.addEventListener("click", (event) => {
      const button = event.target.closest("[data-agent-filter]");
      if (!button) return;
      activeFilter = button.dataset.agentFilter;
      filterBar.querySelectorAll("[data-agent-filter]").forEach((b) => b.classList.toggle("active", b === button));
      calls.forEach((entry) => applyFilterTo(entry.el, entry.agent));
      scroller.scrollTop = scroller.scrollHeight;
    });

    boot();
  }

  document.addEventListener("DOMContentLoaded", initChat);
})(window.MuseAI);
