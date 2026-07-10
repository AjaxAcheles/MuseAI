/**
 * Shared UI primitives for MuseAI.
 *
 * Bootstrap's stylesheet is vendored but its JavaScript bundle is not, so the interactive
 * components below drive Bootstrap's own class names (`.show`, `.active`) directly. That keeps
 * the app to three vendored files and zero network calls.
 *
 * Everything hangs off `window.MuseAI` so the page scripts can share it without a bundler.
 */
"use strict";

window.MuseAI = window.MuseAI || {};

(function (MuseAI) {
  /* ----------------------------------------------------------------- fetch */

  /** POST JSON. Never throws: a transport failure comes back as `{ok:false,error}`. */
  async function postJSON(url, payload) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload === undefined ? {} : payload),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        return { ok: false, error: body.error || `Request failed (${response.status})`, status: response.status };
      }
      return body;
    } catch (error) {
      return { ok: false, error: error.message || "Network request failed" };
    }
  }

  /** GET JSON. Same contract as postJSON. */
  async function getJSON(url) {
    try {
      const response = await fetch(url, { headers: { Accept: "application/json" } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        return { ok: false, error: body.error || `Request failed (${response.status})`, status: response.status };
      }
      return body;
    } catch (error) {
      return { ok: false, error: error.message || "Network request failed" };
    }
  }

  /* ------------------------------------------------------------------ text */

  /**
   * Render untrusted Markdown into an element.
   * Model output is untrusted input, so it is always sanitized before it touches the DOM.
   */
  function renderMarkdownInto(element, text) {
    if (!element) return;
    const source = text || "";
    if (window.marked && window.DOMPurify) {
      element.innerHTML = window.DOMPurify.sanitize(window.marked.parse(source));
    } else {
      element.textContent = source;
    }
  }

  const escapeHtml = (value) => {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    // Quotes too: this helper is used inside attribute values (aria-label,
    // data-tip), where a bare double quote in story text would end the attribute.
    return div.innerHTML.replace(/"/g, "&quot;");
  };

  const formatNumber = (value) => (typeof value === "number" ? value.toLocaleString() : "—");

  /** "1,234 of 5,000 words" style helper that tolerates a missing target. */
  function formatProgress(current, target) {
    if (!target) return `${formatNumber(current || 0)} words`;
    return `${formatNumber(current || 0)} of ${formatNumber(target)} words`;
  }

  /* ------------------------------------------------------------- components */

  /**
   * Tab group. Expects `[data-tab-group]` containing `[data-tab-target="#pane"]` buttons
   * and the panes themselves. Toggles Bootstrap's `.active`/`.show`.
   */
  function initTabs(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-tab-group]").forEach((group) => {
      const buttons = Array.from(group.querySelectorAll("[data-tab-target]"));
      buttons.forEach((button) => {
        button.addEventListener("click", (event) => {
          event.preventDefault();
          buttons.forEach((other) => {
            const selected = other === button;
            other.classList.toggle("active", selected);
            other.setAttribute("aria-selected", selected ? "true" : "false");
            const pane = document.querySelector(other.dataset.tabTarget);
            if (pane) {
              pane.classList.toggle("active", selected);
              pane.classList.toggle("show", selected);
            }
          });
          const pane = document.querySelector(button.dataset.tabTarget);
          if (pane) pane.dispatchEvent(new CustomEvent("tab:shown", { bubbles: true }));
        });
      });
    });
  }

  /**
   * Offcanvas drawer built on Bootstrap's `.offcanvas` styles.
   * Adds a backdrop, traps Escape, and restores focus to the opener.
   */
  function createDrawer(element) {
    if (!element) return null;
    let backdrop = null;
    let opener = null;

    function close() {
      element.classList.remove("show");
      element.setAttribute("aria-hidden", "true");
      if (backdrop) {
        backdrop.remove();
        backdrop = null;
      }
      document.body.classList.remove("drawer-open");
      if (opener) opener.focus();
    }

    function open() {
      opener = document.activeElement;
      backdrop = document.createElement("div");
      backdrop.className = "offcanvas-backdrop fade show";
      backdrop.addEventListener("click", close);
      document.body.appendChild(backdrop);
      document.body.classList.add("drawer-open");
      element.classList.add("show");
      element.setAttribute("aria-hidden", "false");
      const focusable = element.querySelector("input, textarea, button, [tabindex]");
      if (focusable) focusable.focus();
    }

    element.querySelectorAll("[data-drawer-close]").forEach((button) => button.addEventListener("click", close));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && element.classList.contains("show")) close();
    });

    return { open, close, isOpen: () => element.classList.contains("show") };
  }

  /** Transient toast, styled by Bootstrap's `.toast`. */
  function toast(message, variant) {
    const stack = document.getElementById("toast-stack");
    if (!stack) return;
    const node = document.createElement("div");
    node.className = `toast show toast-${variant || "info"}`;
    node.setAttribute("role", "status");
    node.innerHTML = `<div class="toast-body">${escapeHtml(message)}</div>`;
    stack.appendChild(node);
    window.setTimeout(() => {
      node.classList.remove("show");
      window.setTimeout(() => node.remove(), 300);
    }, 5000);
  }

  /** Inline pass/fail line under a form control. */
  function setInlineResult(element, message, ok) {
    if (!element) return;
    element.textContent = message || "";
    element.classList.toggle("is-error", message !== "" && !ok);
    element.classList.toggle("is-ok", message !== "" && Boolean(ok));
  }

  /**
   * Instant hover/focus tooltips for the SVG node maps (dashboard rail, seed
   * timeline). Every element carrying `data-tip` inside `container` gets one;
   * the tip text renders with its newlines. One shared tooltip element serves
   * the whole document, so re-rendering a container just re-attaches listeners
   * to its fresh nodes.
   */
  function attachNodeTooltips(container) {
    if (!container) return;
    let tip = document.getElementById("node-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "node-tooltip";
      tip.className = "node-tooltip";
      tip.hidden = true;
      document.body.appendChild(tip);
      window.addEventListener("scroll", () => (tip.hidden = true), { passive: true });
    }

    const hide = () => (tip.hidden = true);
    container.querySelectorAll("[data-tip]").forEach((node) => {
      const show = () => {
        tip.textContent = node.dataset.tip;
        tip.hidden = false;
        const rect = node.getBoundingClientRect();
        const width = tip.offsetWidth;
        const left = Math.min(Math.max(rect.left + rect.width / 2 - width / 2, 8), window.innerWidth - width - 8);
        let top = rect.top - tip.offsetHeight - 10;
        if (top < 8) top = rect.bottom + 10;
        tip.style.left = `${left}px`;
        tip.style.top = `${top}px`;
      };
      node.addEventListener("mouseenter", show);
      node.addEventListener("mouseleave", hide);
      node.addEventListener("focus", show);
      node.addEventListener("blur", hide);
    });
  }

  /** Standard empty state. Honest absence of data, never a promise of a future feature. */
  function emptyState(title, hint) {
    return `<div class="empty-state">
      <p class="empty-state-title">${escapeHtml(title)}</p>
      ${hint ? `<p class="empty-state-hint">${escapeHtml(hint)}</p>` : ""}
    </div>`;
  }

  MuseAI.postJSON = postJSON;
  MuseAI.getJSON = getJSON;
  MuseAI.renderMarkdownInto = renderMarkdownInto;
  MuseAI.escapeHtml = escapeHtml;
  MuseAI.formatNumber = formatNumber;
  MuseAI.formatProgress = formatProgress;
  MuseAI.initTabs = initTabs;
  MuseAI.createDrawer = createDrawer;
  MuseAI.attachNodeTooltips = attachNodeTooltips;
  MuseAI.toast = toast;
  MuseAI.setInlineResult = setInlineResult;
  MuseAI.emptyState = emptyState;
})(window.MuseAI);
