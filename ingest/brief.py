"""Merge raw request inputs (freeform text, guided fields, extracted doc text)
into a normalized Brief."""
from __future__ import annotations

from typing import Any

from config import LENGTH_PRESETS, DEFAULT_LENGTH
from pipeline.schemas import Brief


def build_brief(form: dict[str, Any], source_excerpt: str = "") -> Brief:
    length = str(form.get("length") or DEFAULT_LENGTH).lower()
    if length not in LENGTH_PRESETS:
        length = DEFAULT_LENGTH
    return Brief(
        idea=str(form.get("idea") or "").strip(),
        source_excerpt=source_excerpt or str(form.get("source_excerpt") or ""),
        genre=str(form.get("genre") or "").strip(),
        tone=str(form.get("tone") or "").strip(),
        pov=str(form.get("pov") or "").strip(),
        length=length,
        characters=str(form.get("characters") or "").strip(),
    )
