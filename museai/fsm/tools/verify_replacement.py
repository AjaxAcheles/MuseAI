"""Check a span rewrite against the splice guard before answering with it.

The reviser's span mode splices the model's rewrite into the draft in place of
the quoted span. A rewrite that balloons or repeats the surrounding prose is
rejected by ``museai/fsm/nodes/revise.py`` and costs a full-beat rewrite; this
tool is that same guard offered up front, so the model can test its own
replacement and fix it instead of losing span mode.
"""

from __future__ import annotations

from typing import Any

VERIFY_REPLACEMENT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "verify_replacement",
        "description": (
            "Check whether a span rewrite is safe to splice into the draft. "
            "Rejects a replacement that grew far beyond the span or that "
            "repeats prose from around the span (which would duplicate it). "
            "Use it in span mode before giving your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "draft": {
                    "type": "string",
                    "description": "The full draft the span sits in.",
                },
                "span": {
                    "type": "string",
                    "description": (
                        "The original span being replaced, quoted exactly as it "
                        "appears in the draft."
                    ),
                },
                "replacement": {
                    "type": "string",
                    "description": "Your candidate rewrite of the span.",
                },
            },
            "required": ["draft", "span", "replacement"],
        },
    },
}


def verify_replacement(draft: str, span: str, replacement: str) -> dict:
    """``{"ok": True}`` when the rewrite splices safely, else the reason it won't."""
    # Imported here, not at module top: the revise node imports the tool
    # registry, so a top-level import back into revise would be a cycle.
    from museai.fsm.nodes.revise import locate, replacement_rejection

    if not (draft or "").strip() or not (replacement or "").strip():
        return {"ok": False, "reason": "draft and replacement must both be non-empty"}

    located = locate(draft, span or "")
    if located is None:
        return {
            "ok": False,
            "reason": (
                "the span was not found in the draft — quote it exactly as it "
                "appears there"
            ),
        }

    reason = replacement_rejection(draft, located, replacement)
    if reason is not None:
        return {"ok": False, "reason": reason}
    return {"ok": True, "reason": "safe to splice"}
