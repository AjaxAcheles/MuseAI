/**
 * Seed parsing, validation, summary, and timeline preview.
 *
 * These helpers are shared by the Seed & Plan page and the Dashboard's Load Seed drawer, so the
 * two surfaces cannot drift apart. Client-side validation mirrors the server's `_validate_seed`
 * in `museai/web/routes/seed.py`; the server remains authoritative and its error wins.
 *
 * The seed schema is exactly: project{id,genre,premise,word_count_target}, arcs[{description}],
 * threads[{description,priority_score,status}], characters[{name,description,pad{...}}].
 * There are no chapters or beats in a seed — the planners create those at run time.
 */
"use strict";

window.MuseAI = window.MuseAI || {};

(function (MuseAI) {
  const escapeHtml = MuseAI.escapeHtml;

  /** Parse seed text. Returns `{ok, value}` or `{ok:false, error}` — never throws. */
  function parseSeedJson(raw) {
    const text = (raw || "").trim();
    if (!text) return { ok: false, error: "Paste or write seed JSON before continuing." };
    try {
      return { ok: true, value: JSON.parse(text) };
    } catch (error) {
      return { ok: false, error: `Malformed JSON: ${error.message}` };
    }
  }

  /** Pretty-print seed text, preserving the original on a parse failure. */
  function formatSeedJson(raw) {
    const parsed = parseSeedJson(raw);
    if (!parsed.ok) return parsed;
    return { ok: true, value: JSON.stringify(parsed.value, null, 2) };
  }

  /**
   * Structural check mirroring the server. Returns `{ok, errors[]}`.
   * Required: project.id (string), a non-empty arcs list, each arc with a description.
   */
  function validateSeedClientSide(seed) {
    const errors = [];
    if (seed === null || typeof seed !== "object" || Array.isArray(seed)) {
      return { ok: false, errors: ["seed must be a JSON object"] };
    }

    const project = seed.project;
    if (!project || typeof project !== "object" || typeof project.id !== "string" || !project.id) {
      errors.push("seed.project.id is required and must be a string");
    }

    if (!Array.isArray(seed.arcs) || seed.arcs.length === 0) {
      errors.push("seed.arcs must be a non-empty list");
    } else {
      seed.arcs.forEach((arc, index) => {
        if (!arc || typeof arc !== "object" || typeof arc.description !== "string" || !arc.description) {
          errors.push(`seed.arcs[${index + 1}].description is required`);
        }
      });
    }

    ["threads", "characters"].forEach((key) => {
      if (key in seed && !Array.isArray(seed[key])) errors.push(`seed.${key} must be a list`);
    });

    (Array.isArray(seed.threads) ? seed.threads : []).forEach((thread, index) => {
      if (!thread || typeof thread.description !== "string" || !thread.description) {
        errors.push(`seed.threads[${index + 1}].description is required`);
      }
    });

    (Array.isArray(seed.characters) ? seed.characters : []).forEach((character, index) => {
      if (!character || typeof character.name !== "string" || !character.name) {
        errors.push(`seed.characters[${index + 1}].name is required`);
      }
      // PAD is optional, but a present PAD must be loadable: the server rejects out-of-range axes.
      if (character && character.pad) {
        ["pleasure", "arousal", "dominance"].forEach((axis) => {
          const value = character.pad[axis];
          if (value === undefined) return;
          if (typeof value !== "number" || Number.isNaN(value)) {
            errors.push(`seed.characters[${index + 1}].pad.${axis} must be a number`);
          } else if (value < -1 || value > 1) {
            errors.push(`seed.characters[${index + 1}].pad.${axis} must be between -1 and 1`);
          }
        });
      }
    });

    return { ok: errors.length === 0, errors };
  }

  /** Counts and title for the success toast and the summary card. */
  function summarizeSeed(seed) {
    const project = (seed && seed.project) || {};
    return {
      title: project.id || "untitled project",
      genre: project.genre || null,
      premise: project.premise || null,
      wordTarget: project.word_count_target || null,
      arcs: Array.isArray(seed && seed.arcs) ? seed.arcs.length : 0,
      threads: Array.isArray(seed && seed.threads) ? seed.threads.length : 0,
      characters: Array.isArray(seed && seed.characters) ? seed.characters.length : 0,
    };
  }

  /** Trim to `max` characters on a word boundary, with an ellipsis when cut. */
  function truncate(text, max) {
    const value = (text || "").trim();
    if (value.length <= max) return value;
    return `${value.slice(0, max).replace(/\s+\S*$/, "")}…`;
  }

  /**
   * Draw the seed as an SVG timeline: a project node, then one node per arc,
   * each captioned with the start of its description and carrying the full
   * description in its hover tooltip. Below the badges, an itemised preview
   * lists every arc, thread, and character the seed declares. Only what the
   * seed actually contains is drawn — no invented chapters or beats.
   */
  function renderSeedTimeline(container, seed) {
    if (!container) return;
    const validation = validateSeedClientSide(seed);
    const arcs = Array.isArray(seed && seed.arcs) ? seed.arcs : [];
    if (!validation.ok && arcs.length === 0) {
      container.innerHTML = MuseAI.emptyState(
        "Nothing to preview yet",
        "Add a project and at least one arc to see the story shape."
      );
      return;
    }

    const summary = summarizeSeed(seed);
    const threads = Array.isArray(seed && seed.threads) ? seed.threads : [];
    const characters = Array.isArray(seed && seed.characters) ? seed.characters : [];
    const gap = 150;
    const left = 70;
    const width = left + Math.max(arcs.length, 1) * gap + 40;
    const y = 60;

    const nodes = arcs
      .map((arc, index) => {
        const cx = left + (index + 1) * gap;
        const label = `Arc ${index + 1}`;
        const description = (arc && arc.description) || "";
        const tip = `${label} of ${arcs.length}\n${description}`;
        return `
          <g class="tl-node" tabindex="0" role="listitem" data-tip="${escapeHtml(tip)}"
             aria-label="${escapeHtml(label)}: ${escapeHtml(description)}">
            <circle class="tl-arc" cx="${cx}" cy="${y}" r="16"></circle>
            <text class="tl-label" x="${cx}" y="${y + 38}" text-anchor="middle">${escapeHtml(label)}</text>
            <text class="tl-snippet" x="${cx}" y="${y + 54}" text-anchor="middle">${escapeHtml(truncate(description, 22))}</text>
          </g>`;
      })
      .join("");

    const endX = left + Math.max(arcs.length, 1) * gap;

    const projectTip = [
      `Project — ${summary.title}`,
      summary.genre ? `Genre: ${summary.genre}` : null,
      summary.wordTarget ? `Target: ${MuseAI.formatNumber(summary.wordTarget)} words` : null,
      summary.premise ? truncate(summary.premise, 220) : null,
    ]
      .filter(Boolean)
      .join("\n");

    const detailGroup = (title, items) =>
      items.length
        ? `<div class="timeline-detail-group">
             <h4>${escapeHtml(title)}</h4>
             <ul>${items.join("")}</ul>
           </div>`
        : "";

    const arcItems = arcs.map(
      (arc, index) => `<li><strong>Arc ${index + 1}</strong> — ${escapeHtml((arc && arc.description) || "")}</li>`
    );
    const threadItems = threads.map((thread) => {
      const meta = [
        thread && thread.status ? `status ${thread.status}` : null,
        thread && typeof thread.priority_score === "number" ? `priority ${thread.priority_score}` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      return `<li>${escapeHtml((thread && thread.description) || "")}${meta ? ` <span class="detail-meta">(${escapeHtml(meta)})</span>` : ""}</li>`;
    });
    const characterItems = characters.map((character) => {
      const pad = character && character.pad;
      const padMeta = pad
        ? ["pleasure", "arousal", "dominance"]
            .filter((axis) => typeof pad[axis] === "number")
            .map((axis) => `${axis[0].toUpperCase()} ${pad[axis]}`)
            .join(" · ")
        : "";
      return `<li><strong>${escapeHtml((character && character.name) || "")}</strong>${
        character && character.description ? ` — ${escapeHtml(truncate(character.description, 120))}` : ""
      }${padMeta ? ` <span class="detail-meta">(${escapeHtml(padMeta)})</span>` : ""}</li>`;
    });

    container.innerHTML = `
      <svg class="timeline" viewBox="0 0 ${width} 130" role="list"
           aria-label="Seed timeline: ${arcs.length} arc${arcs.length === 1 ? "" : "s"}">
        <line class="tl-rail" x1="${left}" y1="${y}" x2="${endX}" y2="${y}"></line>
        <g class="tl-node" tabindex="0" role="listitem" data-tip="${escapeHtml(projectTip)}"
           aria-label="Project ${escapeHtml(summary.title)}">
          <circle class="tl-project" cx="${left}" cy="${y}" r="11"></circle>
          <text class="tl-label" x="${left}" y="${y + 38}" text-anchor="middle">Start</text>
        </g>
        ${nodes}
      </svg>
      <div class="timeline-badges">
        <span class="badge text-bg-secondary">${summary.arcs} arc${summary.arcs === 1 ? "" : "s"}</span>
        <span class="badge text-bg-secondary">${summary.threads} thread${summary.threads === 1 ? "" : "s"}</span>
        <span class="badge text-bg-secondary">${summary.characters} character${summary.characters === 1 ? "" : "s"}</span>
        ${summary.wordTarget ? `<span class="badge text-bg-secondary">${MuseAI.formatNumber(summary.wordTarget)}-word target</span>` : ""}
      </div>
      <div class="timeline-detail">
        ${detailGroup("Arcs", arcItems)}
        ${detailGroup("Threads", threadItems)}
        ${detailGroup("Characters", characterItems)}
      </div>`;
    MuseAI.attachNodeTooltips(container);
  }

  /** POST the seed. The server validates again and its verdict is final. */
  async function submitSeedToBackend(seed) {
    return MuseAI.postJSON("/seed/submit", seed);
  }

  /**
   * Write a plain-text premise into a seed object without disturbing the rest of it.
   * There is no premise-to-outline conversion: the app cannot invent arcs from prose.
   */
  function applyPremise(seed, premise) {
    const next = seed && typeof seed === "object" && !Array.isArray(seed) ? { ...seed } : {};
    next.project = { ...(next.project || {}), premise: premise };
    return next;
  }

  MuseAI.seed = {
    parseSeedJson,
    formatSeedJson,
    validateSeedClientSide,
    summarizeSeed,
    renderSeedTimeline,
    submitSeedToBackend,
    applyPremise,
  };
})(window.MuseAI);
