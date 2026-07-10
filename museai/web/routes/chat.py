"""The View Chat page: every agent's LLM traffic, live and replayed.

Live traffic reaches the browser over the existing `/stream` SSE connection
(`chat_start` / `chat_token` / `chat_end` events published by the LLM client).
This blueprint serves the page itself and the transcript replay that fills in
history on load, so a reload shows the whole conversation, not just calls made
while the tab was open.

Prompts and responses never contain credentials — keys travel in HTTP headers,
which the transcript does not record — so nothing here needs redaction.
"""

from __future__ import annotations

from quart import Blueprint, jsonify, render_template, request

from museai.core import chat_log
from museai.llm.client import live_chat_calls
from museai.web.app import get_config

bp = Blueprint("chat", __name__)

# Every label the engine passes to call_llm, in the order the filter bar shows
# them. "system" is the fallback for a caller that did not name itself.
AGENTS: tuple[str, ...] = (
    "chapter_planner",
    "beat_planner",
    "drafter",
    "critic",
    "reviser",
    "endpoint_test",
    "system",
)

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000


def _clamp_limit(raw: str | None) -> int:
    try:
        value = int(raw) if raw is not None else _DEFAULT_LIMIT
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return max(1, min(value, _MAX_LIMIT))


@bp.get("/chat")
async def chat():
    return await render_template("chat.html", agents=AGENTS)


@bp.get("/chat/history")
async def history():
    limit = _clamp_limit(request.args.get("limit"))
    path = chat_log.default_path(get_config().event_log_path)
    records = chat_log.replay(path, limit)
    # In-flight calls are not on disk yet; without them a page that loads
    # mid-call would show the call empty until it ends, losing every token
    # that streamed before the page arrived.
    return jsonify(
        {
            "ok": True,
            "returned": len(records),
            "records": records,
            "partials": live_chat_calls(),
        }
    )
