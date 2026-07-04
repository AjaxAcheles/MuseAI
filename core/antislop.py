"""Module: M11 (Anti-slop Detection & Correction)

Frozen black-box contract for anti-slop detection and resolution. The two
functions below are the permanent interface the rest of the system builds
against (``node_draft_prose``, M06, calls them unconditionally on every
draft). Programmatically flagging and rewriting the homogenising
"machine-isms" that long-form generation drifts into has no viable approach
yet (see ``_design/conceptual/Open_Problems.md``); this module is an honest
no-op passthrough behind that frozen signature, not a bug or a placeholder
awaiting a forgotten follow-up. Detection may be replaced entirely later
without any caller change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SlopFlag(BaseModel):
    """One flagged span from ``detect_slop``.

    Provisional: the design gives only the two frozen function signatures,
    not a specified finding shape. This model is a best-effort minimal
    contract and may be reshaped behind ``detect_slop``/``resolve_slop``
    once real detection work begins.
    """

    model_config = ConfigDict(extra="forbid")

    offending_text: str  # the flagged span, quoted verbatim — never an integer offset
    reason: str
    suggested_replacement: str | None = None


def detect_slop(text: str) -> list[SlopFlag]:
    """Return no findings until the anti-slop contract is implemented."""
    return []


def resolve_slop(text: str, findings: list[SlopFlag] | None = None) -> str:
    """Return text unchanged until the anti-slop contract is implemented."""
    return text
