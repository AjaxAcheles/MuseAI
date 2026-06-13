"""MuseAI — Quart application.

Serves the single-page UI and the JSON/SSE API that drives the generation
pipeline.
"""
from __future__ import annotations

import json
import os

from quart import Quart, Response, jsonify, render_template, request

from config import LENGTH_PRESETS, settings
from ingest.brief import build_brief
from ingest.extract import extract_text
from export.render import render as render_export
from pipeline.generator import run as run_pipeline
from pipeline.schemas import Brief
from storage import db

app = Quart(__name__)


@app.before_serving
async def _startup() -> None:
    await db.init_db()


@app.get("/")
async def index():
    return await render_template(
        "index.html",
        length_presets=LENGTH_PRESETS,
        has_api_key=settings.has_api_key,
    )


@app.post("/api/projects")
async def create_project():
    form = await request.get_json(force=True, silent=True) or {}
    if not str(form.get("idea") or "").strip():
        return jsonify({"error": "Please describe your idea."}), 400
    brief = build_brief(form)
    pid = await db.create_project(brief.model_dump())
    return jsonify({"id": pid, "brief": brief.model_dump()})


@app.post("/api/projects/<pid>/upload")
async def upload(pid: str):
    project = await db.get_project(pid)
    if project is None:
        return jsonify({"error": "Project not found."}), 404

    files = await request.files
    upload_file = files.get("file")
    if upload_file is None:
        return jsonify({"error": "No file provided."}), 400

    filename = upload_file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_upload_exts:
        return jsonify({"error": f"Unsupported file type: {ext or '(none)'}"}), 400

    data = upload_file.read()
    if len(data) > settings.max_upload_bytes:
        return jsonify({"error": "File is too large."}), 400

    try:
        text = extract_text(filename, data)
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 400

    brief = Brief(**project["brief"])
    # Append to any existing reference material.
    combined = (brief.source_excerpt + "\n\n" + text).strip() if brief.source_excerpt else text
    brief = brief.model_copy(update={"source_excerpt": combined})
    await db.update_brief(pid, brief.model_dump())
    return jsonify({"filename": filename, "chars": len(text), "total_chars": len(combined)})


@app.get("/api/projects/<pid>")
async def fetch_project(pid: str):
    project = await db.get_project(pid)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    return jsonify(project)


@app.get("/api/projects/<pid>/stream")
async def stream(pid: str):
    project = await db.get_project(pid)
    if project is None:
        return jsonify({"error": "Project not found."}), 404

    brief = Brief(**project["brief"])

    async def event_source():
        if not settings.has_api_key:
            yield _sse("error", {"stage": "config", "message": "ANTHROPIC_API_KEY is not set."})
            return
        async for event, data in run_pipeline(pid, brief):
            yield _sse(event, data)

    return Response(
        event_source(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/projects/<pid>/export")
async def export(pid: str):
    fmt = (request.args.get("format") or "md").lower()
    project = await db.get_project(pid)
    if project is None or not project.get("story"):
        return jsonify({"error": "No finished story to export yet."}), 404
    story = project["story"]
    try:
        payload, mimetype = render_export(story["content_md"], fmt)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    safe_title = _slug(story.get("title") or "story")
    return Response(
        payload,
        mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.{fmt}"'},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in text).strip()
    return ("-".join(keep.split()) or "story").lower()[:60]


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port)
