"""Contract tests for the frozen M11 anti-slop interface (core/antislop.py).

These pin the signatures and the honest no-op passthrough behavior. They are
deliberately not detection tests — there is nothing to detect yet (see
_design/conceptual/Open_Problems.md, "Cliché / slop detection has no viable
approach yet"). If real detection ever changes these signatures or the
passthrough guarantee, these tests fail loudly.
"""

import pytest
from pydantic import ValidationError

from core.antislop import SlopFlag, detect_slop, resolve_slop

LONG_STOCK_TEXT = (
    "A shiver ran down her spine as the tapestry of fate unfurled before her. "
    "Little did she know, this was only the beginning of a journey that would "
    "change everything. The weight of the world settled on his shoulders, and "
    "he knew, deep down, that nothing would ever be the same again.\n\n"
    "In the end, it wasn't the destination that mattered, but the journey. "
    "She let out a breath she didn't know she was holding."
)


@pytest.mark.parametrize(
    "text",
    ["", "a", "a short sentence.", LONG_STOCK_TEXT],
)
def test_detect_slop_is_always_empty(text):
    """Pins the honest no-op: detect_slop never flags anything today,
    including deliberately cliché-dense text — this is not a claim that
    the text is clean, only that no detection logic runs yet."""
    assert detect_slop(text) == []


@pytest.mark.parametrize("text", ["", "some prose", LONG_STOCK_TEXT])
def test_resolve_slop_passthrough_no_findings(text):
    assert resolve_slop(text) == text
    assert resolve_slop(text, []) is text or resolve_slop(text, []) == text


def test_resolve_slop_ignores_findings_by_design():
    text = "some prose that could theoretically be flagged"
    finding = SlopFlag(offending_text="prose", reason="test finding")
    assert resolve_slop(text, [finding]) == text


def test_slopflag_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        SlopFlag(offending_text="x", reason="y", offset=3)


def test_slopflag_requires_verbatim_span_field():
    with pytest.raises(ValidationError):
        SlopFlag(reason="missing offending_text")


def test_slopflag_never_carries_integer_offsets():
    flag = SlopFlag(offending_text="x", reason="y")
    assert isinstance(flag.offending_text, str)
    assert not hasattr(flag, "offset")
    assert not hasattr(flag, "start")
    assert not hasattr(flag, "end")


def test_contract_accepts_partial_or_complete_text():
    partial = "this sentence is cut off mid-w"
    complete = "this sentence is complete."
    assert detect_slop(partial) == []
    assert detect_slop(complete) == []
    assert resolve_slop(partial) == partial
    assert resolve_slop(complete) == complete


def test_node_draft_prose_imports_core_antislop():
    import fsm.nodes.node_draft_prose as node_draft_prose

    assert node_draft_prose.detect_slop is detect_slop
    assert node_draft_prose.resolve_slop is resolve_slop
