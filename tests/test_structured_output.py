"""Module: M04 (LLM Inference Boundary)
Synthetic tests for provider-neutral structured-output helpers.

These exercise the validate -> lenient-salvage path on text and a local
synthetic Pydantic model only. No endpoint, no real model, no network, and no
retry loop is involved here; the bounded retry cap lives in the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, ConfigDict

import llm.call_llm as call_llm_module
from llm.call_llm import (
    LLMCallError,
    StructuredOutputError,
    UnsupportedGrammarStrategyError,
    call_llm_structured,
    extract_first_json_object,
    validate_structured_text,
    validate_with_salvage,
)


class _SyntheticRecord(BaseModel):
    """Local synthetic schema standing in for any strict LLM-returned model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_code: str
    offending_text: str


def _valid_payload() -> dict:
    return {"error_code": "PACING_ISSUE", "offending_text": "too fast"}


def test_strict_valid_json_validates() -> None:
    record = validate_structured_text(json.dumps(_valid_payload()), _SyntheticRecord)
    assert isinstance(record, _SyntheticRecord)
    assert record.error_code == "PACING_ISSUE"
    assert record.offending_text == "too fast"


def test_invalid_json_fails() -> None:
    # Strict validation of unparseable text raises a clear structured error.
    with pytest.raises(StructuredOutputError):
        validate_structured_text("not json at all", _SyntheticRecord)

    # Parseable JSON that violates the schema (unknown key) also fails strictly.
    bad_schema = json.dumps({**_valid_payload(), "bogus": 1})
    with pytest.raises(StructuredOutputError):
        validate_structured_text(bad_schema, _SyntheticRecord)


def test_fenced_json_validates_through_salvage() -> None:
    fenced = f"```json\n{json.dumps(_valid_payload())}\n```"
    # Strict validation rejects the fenced wrapper...
    with pytest.raises(StructuredOutputError):
        validate_structured_text(fenced, _SyntheticRecord)
    # ...but salvage strips the fence and recovers the object.
    record = validate_with_salvage(fenced, _SyntheticRecord)
    assert record.error_code == "PACING_ISSUE"


def test_prose_wrapped_json_validates_through_salvage() -> None:
    prose = (
        "Sure! Here is the failure record you asked for:\n"
        f"{json.dumps(_valid_payload())}\n"
        "Let me know if you need anything else."
    )
    with pytest.raises(StructuredOutputError):
        validate_structured_text(prose, _SyntheticRecord)
    record = validate_with_salvage(prose, _SyntheticRecord)
    assert record.offending_text == "too fast"


def test_malformed_braces_fail_cleanly() -> None:
    # An unbalanced object cannot be extracted and salvage fails (no hang).
    malformed = '{"error_code": "PACING_ISSUE", "offending_text": "too fast"'
    assert extract_first_json_object(malformed) is None
    with pytest.raises(StructuredOutputError):
        validate_with_salvage(malformed, _SyntheticRecord)


def test_extraction_respects_braces_inside_quoted_strings() -> None:
    # The offending_text value itself contains braces and an escaped quote; the
    # extractor must return the whole object, not stop at the inner brace.
    payload = {
        "error_code": "STYLE",
        "offending_text": 'the phrase "a {curly} aside" with a \\" inside',
    }
    wrapped = f"prefix text {json.dumps(payload)} trailing text"

    extracted = extract_first_json_object(wrapped)
    assert extracted is not None
    assert json.loads(extracted) == payload

    record = validate_with_salvage(wrapped, _SyntheticRecord)
    assert record.error_code == "STYLE"
    assert "{curly}" in record.offending_text


# --------------------------------------------------------------------------- #
# call_llm_structured: bounded retry / strategy selection (call-level injection)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FakeEndpoint:
    base_url: str = "https://endpoint.example.test/v1"
    grammar_constraint_strategy: str = "gbnf"


@dataclass
class _FakeResponse:
    text: str


@dataclass
class _RecordingCall:
    """Async stand-in for ``call_llm`` that returns queued texts in order."""

    texts: list[str]
    calls: list[dict] = field(default_factory=list)
    index: int = 0

    async def __call__(self, messages, endpoint, **kwargs):
        self.calls.append(kwargs)
        text = self.texts[min(self.index, len(self.texts) - 1)]
        self.index += 1
        return _FakeResponse(text=text)


@pytest.fixture(autouse=True)
def _silence_io_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the salvage log path off the real logs/ file during tests.
    handlerless = logging.getLogger("test_structured_output_io")
    handlerless.handlers = []
    monkeypatch.setattr(
        call_llm_module.llm_io_logger, "get_llm_io_logger", lambda: handlerless
    )


def test_structured_returns_validated_object_on_first_valid() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    record = asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(),
            schema_model=_SyntheticRecord,
            validate_retry_cap=3,
            call=call,
        )
    )
    assert isinstance(record, _SyntheticRecord)
    assert len(call.calls) == 1  # validated immediately, no extra attempts


def test_structured_retries_until_cap_then_succeeds() -> None:
    # Two invalid responses, then a valid one — within the cap of 3.
    call = _RecordingCall(
        texts=["not json", "{still bad", json.dumps(_valid_payload())]
    )
    record = asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(),
            schema_model=_SyntheticRecord,
            validate_retry_cap=3,
            call=call,
        )
    )
    assert record.error_code == "PACING_ISSUE"
    assert len(call.calls) == 3  # retried exactly up to the cap


def test_structured_salvages_final_response_after_cap() -> None:
    # Every attempt is wrapper-fenced (strict-invalid); the final one salvages.
    fenced = f"```json\n{json.dumps(_valid_payload())}\n```"
    call = _RecordingCall(texts=[fenced])
    record = asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(),
            schema_model=_SyntheticRecord,
            validate_retry_cap=2,
            call=call,
        )
    )
    assert record.offending_text == "too fast"
    assert len(call.calls) == 2  # exhausted the cap, then salvaged once


def test_structured_raises_hard_failure_when_unsalvageable() -> None:
    call = _RecordingCall(texts=["totally not json and no braces"])
    with pytest.raises(LLMCallError) as excinfo:
        asyncio.run(
            call_llm_structured(
                [{"role": "user", "content": "go"}],
                _FakeEndpoint(),
                schema_model=_SyntheticRecord,
                validate_retry_cap=2,
                call=call,
            )
        )
    assert excinfo.value.attempt_count == 2
    assert len(call.calls) == 2


def test_structured_gbnf_strategy_injects_grammar_option() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(grammar_constraint_strategy="gbnf"),
            schema_model=_SyntheticRecord,
            validate_retry_cap=1,
            call=call,
        )
    )
    sent = call.calls[0]
    assert sent["response_format"] is None
    assert "grammar" in sent["extra_body"]
    assert isinstance(sent["extra_body"]["grammar"], str) and sent["extra_body"]["grammar"]


def test_structured_json_mode_strategy_sets_response_format() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(grammar_constraint_strategy="json_mode"),
            schema_model=_SyntheticRecord,
            validate_retry_cap=1,
            call=call,
        )
    )
    sent = call.calls[0]
    assert sent["response_format"] == {"type": "json_object"}


def test_structured_json_schema_strategy_sets_response_format() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    asyncio.run(
        call_llm_structured(
            [{"role": "user", "content": "go"}],
            _FakeEndpoint(grammar_constraint_strategy="json_schema"),
            schema_model=_SyntheticRecord,
            validate_retry_cap=1,
            call=call,
        )
    )
    sent = call.calls[0]
    rf = sent["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == _SyntheticRecord.__name__
    assert rf["json_schema"]["strict"] is True
    assert isinstance(rf["json_schema"]["schema"], dict) and rf["json_schema"]["schema"]


def test_structured_unknown_strategy_raises_config_error() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    with pytest.raises(UnsupportedGrammarStrategyError):
        asyncio.run(
            call_llm_structured(
                [{"role": "user", "content": "go"}],
                _FakeEndpoint(grammar_constraint_strategy="mystery_mode"),
                schema_model=_SyntheticRecord,
                validate_retry_cap=2,
                call=call,
            )
        )
    assert len(call.calls) == 0  # rejected before any endpoint call


def test_structured_rejects_nonpositive_cap() -> None:
    call = _RecordingCall(texts=[json.dumps(_valid_payload())])
    with pytest.raises(ValueError):
        asyncio.run(
            call_llm_structured(
                [{"role": "user", "content": "go"}],
                _FakeEndpoint(),
                schema_model=_SyntheticRecord,
                validate_retry_cap=0,
                call=call,
            )
        )
