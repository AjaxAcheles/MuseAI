"""Shared deterministic validation for chapter-plan obligations."""

from __future__ import annotations


_PLACEHOLDER_OBLIGATIONS = frozenset(
    {
        "an event that must occur",
        "a concrete event that must occur",
        "a concrete event, reveal, or end state",
        "another",
        "a concrete event or condition that must exist",
        "a moment of tension or conflict that must occur",
        "the climax of the story where the outcome is determined",
        "a resolution or conclusion to the story",
    }
)


def validate_concrete_obligation(value: object) -> tuple[str | None, str | None]:
    """Return normalized text and a deterministic quality problem, if any.

    The rejected phrases are output-format placeholders observed in planner
    replies. This intentionally does not attempt subjective prose evaluation:
    story-specific but broad obligations remain valid.
    """
    if not isinstance(value, str):
        return None, f"must be a string, got {type(value).__name__}"

    normalized = " ".join(value.split())
    if not normalized:
        return None, "must be a non-empty string"
    if normalized.casefold() in _PLACEHOLDER_OBLIGATIONS:
        return None, (
            f"repeats the output-format placeholder {normalized!r}; replace it with "
            "a concrete story event, reveal, or end state"
        )
    return normalized, None