"use strict";

const $ = (sel) => document.querySelector(sel);

let projectId = null;
let evtSource = null;

// ---- view switching --------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("#view-" + name).classList.add("active");
}

// ---- file upload (deferred until generate) --------------------------------
const fileInput = $("#file");
const dropzone = $("#dropzone");
$("#browse").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    $("#file-status").textContent = "Selected: " + fileInput.files[0].name;
  }
});
["dragover", "dragenter"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (ev) => {
  if (ev.dataTransfer.files.length) {
    fileInput.files = ev.dataTransfer.files;
    $("#file-status").textContent = "Selected: " + fileInput.files[0].name;
  }
});

// ---- submit brief ----------------------------------------------------------
$("#brief-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("#brief-error").textContent = "";
  const btn = $("#generate-btn");
  btn.disabled = true;
  btn.textContent = "Starting…";

  const brief = {
    idea: $("#idea").value,
    genre: $("#genre").value,
    tone: $("#tone").value,
    pov: $("#pov").value,
    length: $("#length").value,
    characters: $("#characters").value,
  };

  try {
    const res = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(brief),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not create project.");
    projectId = data.id;

    if (fileInput.files.length) {
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      const up = await fetch(`/api/projects/${projectId}/upload`, { method: "POST", body: fd });
      const uj = await up.json();
      if (!up.ok) throw new Error(uj.error || "Upload failed.");
    }

    startGeneration();
  } catch (e) {
    $("#brief-error").textContent = e.message;
    btn.disabled = false;
    btn.textContent = "Generate story";
  }
});

// ---- generation (SSE) ------------------------------------------------------
let proseTarget = null;

function resetTimeline() {
  document.querySelectorAll("#timeline li").forEach((li) => {
    li.classList.remove("active", "done");
    li.querySelector(".detail").textContent = "";
  });
  $("#live-prose").innerHTML = "";
  $("#live-title").textContent = "Writing…";
  $("#generate-error").textContent = "";
  proseTarget = null;
}

function markStage(stage, state) {
  const li = document.querySelector(`#timeline li[data-stage="${stage}"]`);
  if (!li) return;
  if (state === "active") {
    document.querySelectorAll("#timeline li.active").forEach((x) => {
      x.classList.remove("active"); x.classList.add("done");
    });
    li.classList.add("active");
  } else if (state === "done") {
    li.classList.remove("active"); li.classList.add("done");
  }
}

function setDetail(stage, text) {
  const li = document.querySelector(`#timeline li[data-stage="${stage}"]`);
  if (li) li.querySelector(".detail").textContent = text;
}

function startGeneration() {
  resetTimeline();
  showView("generate");

  evtSource = new EventSource(`/api/projects/${projectId}/stream`);

  evtSource.addEventListener("stage_start", (e) => {
    const d = JSON.parse(e.data);
    markStage(d.stage, "active");
    if (d.stage === "revise") {
      $("#live-title").textContent = "Polishing the final draft…";
      const tag = document.createElement("div");
      tag.className = "scene-tag";
      tag.textContent = "— Final polish —";
      $("#live-prose").appendChild(tag);
      proseTarget = document.createElement("div");
      proseTarget.className = "chunk";
      $("#live-prose").appendChild(proseTarget);
    }
  });

  evtSource.addEventListener("stage_data", (e) => {
    const d = JSON.parse(e.data);
    const p = d.payload;
    if (d.stage === "premise") setDetail("premise", `${p.title} — ${p.logline}`);
    if (d.stage === "bible") setDetail("bible", `${p.characters.length} characters · ${p.pov}`);
    if (d.stage === "outline") setDetail("outline", `${p.scenes.length} scenes mapped`);
    markStage(d.stage, "done");
  });

  evtSource.addEventListener("scene_start", (e) => {
    const d = JSON.parse(e.data);
    setDetail("draft", `Scene ${d.index} of ${d.total}: ${d.title}`);
    $("#live-title").textContent = "Writing the scenes…";
    const tag = document.createElement("div");
    tag.className = "scene-tag";
    tag.textContent = `Scene ${d.index}: ${d.title}`;
    $("#live-prose").appendChild(tag);
    proseTarget = document.createElement("div");
    proseTarget.className = "chunk";
    $("#live-prose").appendChild(proseTarget);
  });

  evtSource.addEventListener("draft_delta", (e) => appendProse(JSON.parse(e.data).text));
  evtSource.addEventListener("revise_delta", (e) => appendProse(JSON.parse(e.data).text));

  evtSource.addEventListener("scene_done", () => {});

  evtSource.addEventListener("done", (e) => {
    markStage("revise", "done");
    evtSource.close();
    loadStory();
  });

  evtSource.addEventListener("error", (e) => {
    let msg = "Generation failed. Check the server logs.";
    // EventSource also fires "error" with no data on a dropped connection.
    if (e.data) {
      try {
        const d = JSON.parse(e.data);
        msg = d.stage ? `[${d.stage}] ${d.message || msg}` : (d.message || msg);
        console.error("MuseAI generation error:", d.error_type || "", d.message || "", "\n", d.trace || "");
      } catch (_) {}
    }
    $("#generate-error").textContent = msg;
    if (evtSource) evtSource.close();
  });
}

function appendProse(text) {
  if (!proseTarget) return;
  proseTarget.textContent += text;
  const pane = $("#live-prose");
  pane.scrollTop = pane.scrollHeight;
}

// ---- reader ----------------------------------------------------------------
async function loadStory() {
  const res = await fetch(`/api/projects/${projectId}`);
  const data = await res.json();
  if (!data.story) {
    $("#generate-error").textContent = "Story finished but could not be loaded.";
    return;
  }
  renderStory(data.story.content_md);
  for (const fmt of ["md", "txt", "docx", "pdf"]) {
    $("#exp-" + fmt).href = `/api/projects/${projectId}/export?format=${fmt}`;
  }
  showView("reader");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderStory(md) {
  const lines = md.split("\n");
  let title = "Untitled";
  let bodyStart = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("# ")) { title = lines[i].slice(2).trim(); bodyStart = i + 1; break; }
  }
  const body = lines.slice(bodyStart).join("\n").trim();
  const blocks = body.split(/\n\s*\n/);
  const art = $("#story");
  art.innerHTML = "";
  const h1 = document.createElement("h1");
  h1.textContent = title;
  art.appendChild(h1);
  let firstPara = true;
  for (const raw of blocks) {
    const block = raw.trim();
    if (!block) continue;
    if (block === "* * *" || block === "***" || block === "---") {
      const sb = document.createElement("p");
      sb.className = "scenebreak";
      sb.textContent = "* * *";
      art.appendChild(sb);
      firstPara = true;
      continue;
    }
    const p = document.createElement("p");
    if (firstPara) { p.className = "first"; firstPara = false; }
    p.textContent = block.replace(/\n/g, " ");
    art.appendChild(p);
  }
}

$("#new-story").addEventListener("click", () => {
  projectId = null;
  $("#brief-form").reset();
  $("#file-status").textContent = "";
  $("#generate-btn").disabled = false;
  $("#generate-btn").textContent = "Generate story";
  showView("brief");
});
