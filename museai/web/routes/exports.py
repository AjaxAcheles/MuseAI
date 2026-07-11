"""Manuscript export routes.

Thin wrappers over :func:`museai.fsm.export.export_manuscript`, which already assembles committed
beat prose in narrative order. Nothing is re-implemented here; the run-completion path and this
button produce the same file.
"""

from __future__ import annotations

from pathlib import Path

from quart import Blueprint, jsonify, render_template, send_file

from museai.fsm.export import committed_word_count, export_manuscript
from museai.web.app import get_config
from museai.web.routes.dashboard import _active_project_id

bp = Blueprint("exports", __name__)


def _manuscript_path() -> Path | None:
    """Where `export_manuscript` writes for the running (else seeded) project."""
    project_id = _active_project_id()
    if project_id is None:
        return None
    return Path("data/output") / f"{project_id}.md"


@bp.get("/exports")
async def exports():
    path = _manuscript_path()
    existing = None
    if path is not None and path.is_file():
        stat = path.stat()
        existing = {"path": str(path), "bytes": stat.st_size, "modified_at": stat.st_mtime}
    return await render_template(
        "exports.html",
        existing=existing,
        word_count=committed_word_count(get_config(), project_id=_active_project_id()),
    )


@bp.post("/exports/manuscript")
async def manuscript():
    cfg = get_config()
    project_id = _active_project_id()
    word_count = committed_word_count(cfg, project_id=project_id)
    if project_id is None or word_count == 0:
        # Writing an empty file would look like a successful export of nothing.
        return jsonify({"ok": False, "error": "No committed story to export yet."}), 400

    path = export_manuscript(cfg, project_id=project_id)
    return jsonify({"ok": True, "path": str(path), "word_count": word_count})


@bp.get("/exports/download")
async def download():
    path = _manuscript_path()
    if path is None or not path.is_file():
        return jsonify({"ok": False, "error": "No manuscript has been exported yet."}), 404
    return await send_file(path, mimetype="text/markdown", as_attachment=True, attachment_filename=path.name)
