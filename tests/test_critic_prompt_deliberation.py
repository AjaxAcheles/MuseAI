"""What the critic prompt asks the model to *think about*, measured.

This file changes nothing. It exists because the fix it belongs to — trimming
the continuity critic's deliberation surface — was deliberately not taken, and a
prompt change is the kind of edit that is easy to make and impossible to
evaluate after the fact.

The problem it measures: on the 2026-07-25 live run the critic spent its entire
4,096-token completion grant reasoning and returned zero characters on 30 of 81
calls — 30.2 minutes, 59% of all critic wall clock. The truncated calls were not
near misses. Every one of the 51 calls that *answered* fit inside the cap (the
largest used 3,919 tokens); every one of the 30 that failed was still
deliberating when it hit the ceiling, with 18,216–23,284 characters of reasoning
against a median of 7,441 for the answered ones. Two cleanly separated
populations, which is why raising the cap from 2,048 to 4,096 left the failure
rate at exactly 37% and merely doubled the price of each failure.

The suspects are in the prompt: an eight-category report list, and a section
naming two contradictions as "worth checking deliberately". Both are reasonable
instructions that a reasoning model can expand indefinitely.

So this file does two things:

* **Pins the surface.** The counts below are the current shape of the prompt.
  A trim is supposed to move them, and a number moving in a test diff is a
  change someone chose; a prompt quietly growing a ninth category is not.
* **Prices the trim without making it.** `test_pricing_*` renders the real
  prompt, excises a candidate section from the rendered text, and reports what
  that would have saved — the measurement a decision to cut it needs, taken
  against the real template rather than an estimate.

What it does not do is assert the trim happened. If the sections are removed
these tests fail loudly, and that is the point: someone then re-reads the
numbers and re-pins them, having seen the cost.
"""

from __future__ import annotations

import pytest

from museai.fsm.nodes.context_budget import window_budget
from museai.llm.prompts import render_messages
from museai.llm.tokenizer import count_message_tokens

# The two passages under suspicion, identified by their opening words so the
# test names the prose rather than a line number that drifts.
DELIBERATE_CHECK_ANCHOR = "Two contradictions are easy to miss"
REPORT_LIST_ANCHOR = "Report a problem only when the draft actually contradicts"

# The current shape. These are observations, not targets.
EXPECTED_ERROR_CODES = 7
EXPECTED_REPORT_BULLETS = 8
EXPECTED_TOOL_BULLETS = 7


# A context package sized like a real mid-chapter beat rather than a fixture
# stub: the live prompts this is calibrated against measured 3,138–7,535 tokens,
# and a two-sentence draft would price the trim against nothing.
DRAFT = " ".join(
    [
        "The kettle clicked the burner off somewhere inside.",
        "Nell set the basket down on the step and did not go in.",
        "The gate sat unlatched, the way Ida never bothered to fix.",
    ]
    * 12
)

PACKAGE = {
    "beat": {
        "id": "arc-1-c02-b02",
        "intent": "Nell decides whether to say anything about the ladder.",
        "required_change": "Nell moves from avoidance to a decision.",
        "observable_event": "She sets the basket down and stays outside.",
        "entry_state": "Nell is carrying the basket up the path.",
        "exit_state": "Nell has decided to tell Ida the truth.",
        "discharges": ["Establish the debt between them."],
        "thread_updates": [{"id": "thread-ladder", "status": "advanced"}],
    },
    "project": {
        "genre": "quiet literary fiction",
        "premise": "A borrowed ladder becomes a debt neither woman will name.",
        "setting": "Two adjoining gardens in a coastal town.",
    },
    "chapter": {
        "id": "arc-1-c02",
        "obligations": [
            "Establish the debt between them.",
            "Show the gate as a shared boundary.",
        ],
    },
    "threads": [
        {"id": "thread-ladder", "status": "open", "description": "The unreturned ladder."},
        {"id": "thread-gate", "status": "open", "description": "The gate neither will latch."},
    ],
    "characters": [
        {"id": "char-nell", "name": "Nell", "description": "Avoids confrontation; keeps accounts in her head."},
        {"id": "char-ida", "name": "Ida", "description": "Talks about weather to avoid talking about anything."},
    ],
    "recent_prose": [
        "She had carried the basket up that path a hundred times." * 8,
        "Ida had said nothing about the ladder for three weeks." * 8,
    ],
}


def _render(repetition_overlap_count: int = 0):
    return render_messages(
        "continuity_critic",
        draft_text=DRAFT,
        beat=PACKAGE["beat"],
        project=PACKAGE["project"],
        chapter=PACKAGE["chapter"],
        threads=PACKAGE["threads"],
        characters=PACKAGE["characters"],
        recent_prose=PACKAGE["recent_prose"],
        research_mode=False,
        repetition_overlap_count=repetition_overlap_count,
    )


@pytest.fixture
def critic_messages():
    return _render()


@pytest.fixture
def endpoint(config_factory):
    # `config_factory` leaves `context_window` unset, which makes `window_budget`
    # return None and the fit assertion vacuous. These two numbers are the shape
    # `config.yaml` gives the critic today; they are test data standing in for a
    # configured endpoint, not a second copy of the setting.
    return config_factory().endpoint_for("critic").model_copy(
        update={"context_window": 16384, "output_reservation": 4096}
    )


def _tokens(messages, endpoint):
    return count_message_tokens(messages, endpoint.tokenizer_family, endpoint.model_name)


def _system(messages):
    return next(m["content"] for m in messages if m["role"] == "system")


def _excise(messages, start_anchor, end_anchor=None):
    """The same prompt with one section removed, for pricing a cut."""
    out = []
    for message in messages:
        if message["role"] != "system":
            out.append(dict(message))
            continue
        text = message["content"]
        start = text.index(start_anchor)
        end = text.index(end_anchor) if end_anchor else len(text)
        out.append({**message, "content": text[:start] + text[end:]})
    return out


# ------------------------------------------------------------ the surface


def test_the_deliberate_check_section_is_still_present(critic_messages):
    """The passage that names two contradictions as worth checking deliberately.
    It is there on purpose — it was added to catch a character mourned in one
    chapter picking up the phone in the next — and it is also the most direct
    invitation in the prompt to think longer."""
    assert DELIBERATE_CHECK_ANCHOR in _system(critic_messages)


def test_the_report_list_has_not_quietly_grown(critic_messages):
    """Eight categories of problem to look for. Each is a separate judgement the
    model has to make about the whole draft before it can answer."""
    system = _system(critic_messages)
    report_section = system[system.index(REPORT_LIST_ANCHOR):]
    bullets = [
        line for line in report_section.splitlines() if line.strip().startswith("- ")
    ]
    assert len(bullets) == EXPECTED_REPORT_BULLETS


@pytest.mark.parametrize("repetition_overlap_count", [0, 2])
def test_the_tool_roster_in_the_prompt_matches_the_real_one(repetition_overlap_count):
    """A prompt that offers a tool the registry does not serve teaches the model
    to invent names. The live run fabricated four — `check_continuity`,
    `check_draft_against_context`, `check_draft_for_continuity`, and one that
    parsed as `[]` — so the two lists have to agree.

    Parametrized over both repetition-check branches: `find_repetition` was
    dropped from the roster specifically because a hit had nowhere to be
    reported (see the error-code test below), and that has to hold whether or
    not the audit actually found an overlap on this draft."""
    from museai.fsm.tools.registry import AGENT_TOOLS

    system = _system(_render(repetition_overlap_count))
    offered = {
        line.strip()[2:].split(":")[0].strip()
        for line in system.splitlines()
        if line.strip().startswith("- ") and ":" in line
    }
    roster = set(AGENT_TOOLS["critic"])
    assert roster <= offered, f"prompt omits real tools: {roster - offered}"
    assert len(roster) == EXPECTED_TOOL_BULLETS
    assert "find_repetition" not in system


@pytest.mark.parametrize("repetition_overlap_count", [0, 2])
def test_every_named_error_code_is_one_the_parser_accepts(repetition_overlap_count):
    """A code the model is told to use but the parser rejects is a re-prompt the
    critic can never win — it would answer correctly and be told it was wrong.
    The two lists have to be the same list.

    Parametrized over both repetition-check branches on purpose: an earlier
    version of the overlap-found block spelled its status as the literal token
    `PARAGRAPH_OVERLAP`, which this test would have caught immediately had it
    run against a nonzero count — the fixture only ever rendered the clean
    branch, so a clean suite proved nothing about the branch that actually
    ships on a draft with real overlaps. That token collided with
    `CRITIC_ERROR_CODES` exactly the way this test exists to prevent: a model
    that echoed it back as an `error_code` would hit `StructuredOutputError`
    on every repeat-heavy draft, feeding the same unreadable-verdict retry
    loop F-A and F-D closed."""
    from museai.llm.structured import CRITIC_ERROR_CODES

    prompt = " ".join(m["content"] for m in _render(repetition_overlap_count))
    named = {
        word.strip(',."')
        for word in prompt.split()
        if word.strip(',."').isupper() and "_" in word.strip(',."')
    }
    assert named == set(CRITIC_ERROR_CODES)
    assert len(named) == EXPECTED_ERROR_CODES


def test_a_clean_draft_is_told_the_repetition_check_passed():
    user = _render(0)[1]["content"]
    assert '<repetition_check status="clean"/>' in user


def test_an_overlap_carries_the_count_and_a_do_not_repeat_instruction():
    """The instruction is the only guard against double-counting: `total` in
    `adversarial_critics` sums `state["critic_failures"]` (already carrying
    audit's own overlap findings) with whatever the critic reports, and
    nothing in code deduplicates a critic finding that restates one of them."""
    messages = _render(2)
    user = messages[1]["content"]
    system = messages[0]["content"]
    assert 'status="overlap_found" count="2"' in user
    assert "do not report" in user.lower()
    assert "do not add your own finding" in system.lower()


# ------------------------------------------------------------ pricing a cut


def test_pricing_the_deliberate_check_section(critic_messages, endpoint):
    """What removing the "worth checking deliberately" passage would save.

    Reported, not enforced. The saving is in *prompt* tokens, and the failure
    this is aimed at is on the output side — so a small number here is itself
    the finding: it would mean the section earns its place cheaply and the
    reason to cut it would have to be its effect on reasoning length, which
    only a live run can measure (watch `empty_reply` in fsm.log).
    """
    full = _tokens(critic_messages, endpoint)
    trimmed = _tokens(
        _excise(critic_messages, DELIBERATE_CHECK_ANCHOR, REPORT_LIST_ANCHOR), endpoint
    )
    saved = full - trimmed
    print(f"\ndeliberate-check section: {saved} tokens of {full} ({100 * saved / full:.1f}%)")
    assert 0 < saved < full


def test_pricing_the_whole_instruction_block(critic_messages, endpoint):
    """The ceiling on what any prompt trim can buy: everything from the report
    list to the end of the system message."""
    full = _tokens(critic_messages, endpoint)
    trimmed = _tokens(_excise(critic_messages, REPORT_LIST_ANCHOR), endpoint)
    saved = full - trimmed
    print(f"\ninstruction block: {saved} tokens of {full} ({100 * saved / full:.1f}%)")
    assert 0 < saved < full


def test_the_prompt_leaves_room_for_the_verdict(critic_messages, endpoint):
    """The property any trim must preserve, and the one the live failures were
    wrongly blamed on. Prompt size was never the binding constraint — measured
    prompts ran 3,138–7,535 tokens against a 12,288 budget and
    `retry_context_dropped` never once fired. A trim must not make this
    assertion the interesting one."""
    budget = window_budget(endpoint)
    assert budget is not None
    assert _tokens(critic_messages, endpoint) <= budget
