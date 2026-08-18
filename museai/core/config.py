"""Strict configuration loading for MuseAI v1.

Config is loaded from ``config.yaml`` into Pydantic v2 models. Every model
forbids extra keys, so an unknown or mistyped key is fatal at boot. Thresholds,
caps, and endpoint details all live in config — nothing here is hardcoded.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Load .env once at import so ${VAR} references can be resolved.
load_dotenv()

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


# The five LLM-powered agent roles. Every ``agents:`` key must be one of these.
AGENT_ROLES = ("chapter_planner", "beat_planner", "drafter", "reviser", "critic")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}


def _resolve_env_api_key(value: str) -> str:
    """Resolve a ``${VAR}`` api_key from the environment (via .env)."""
    match = _ENV_REF.match(value.strip())
    if not match:
        return value
    var = match.group(1)
    resolved = os.environ.get(var)
    if resolved is None or resolved == "":
        raise ValueError(
            f"api_key references environment variable ${{{var}}}, "
            f"but it is unset or empty"
        )
    return resolved


def _validate_reasoning_effort(value: str | None) -> str | None:
    """Reject reasoning-budget hints an OpenAI-compatible server cannot name."""
    if value is not None and value not in _REASONING_EFFORTS:
        accepted = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(
            f"reasoning_effort must be one of {accepted}, got {value!r}"
        )
    return value


class EndpointConfig(BaseModel):
    """LLM endpoint description.

    Endpoint/model-agnostic on purpose: no provider, model, or port names appear
    in logic — everything the adapter needs is here.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str
    model_name: str
    tokenizer_family: Literal["tiktoken", "char_heuristic"]
    request_timeout: int = 60
    # Maximum gap between streamed chunks, separate from request_timeout (which
    # governs connect/write/pool). A reasoning endpoint legitimately pauses far
    # longer between tokens than it takes to open a connection.
    stream_read_timeout: int = 300
    temperature: float = 0.7
    # OpenAI-compatible hint for the endpoint's hidden reasoning budget. None
    # omits it entirely, preserving existing requests; "none" disables hidden
    # reasoning where honoured. A server may silently ignore unknown fields:
    # Ollama's /v1 layer honours reasoning_effort but ignores think (0.32.1).
    reasoning_effort: str | None = None
    # Sent as max_tokens when set. None omits the field entirely, preserving the
    # v1.02 wire-format finding: some endpoints bill hidden reasoning against
    # this budget. Truncation is detected via finish_reason either way.
    max_output_tokens: int | None = None
    # Merged verbatim into every request body. The endpoint-agnostic escape hatch
    # for server-specific knobs the OpenAI shape has no field for — notably
    # Ollama's context window: extra_body={"options": {"num_ctx": 16384}}.
    #
    # Be aware that a server is free to ignore what it does not recognise, and
    # silently: Ollama's /v1 compatibility layer drops `options` entirely (verified
    # against 0.32.1), so num_ctx there is decoration, not configuration. This is
    # why served_prompt_tokens is checked against context_window at the call site —
    # an escape hatch you cannot verify is an escape hatch you cannot trust.
    extra_body: dict[str, Any] | None = None
    # Ask the endpoint to append a usage block to its stream (the OpenAI-documented
    # `stream_options.include_usage`). Non-streamed replies carry usage unasked;
    # streamed ones do not, and without it a streaming-only deployment can never
    # learn the server's real token counts. Default on because both OpenAI and
    # Ollama honour it; set false for an endpoint that 400s on the unknown key.
    stream_usage: bool = True
    # The model's real context window, in tokens. When set, MuseAI trims assembled
    # prompts (planners and drafting) to leave output_reservation tokens free, so
    # a small window never leaves zero room to generate. None = trust the endpoint
    # and do not trim (large models). Endpoint-agnostic: not tied to Ollama.
    context_window: int | None = None
    # Minimum tokens kept free for the model to generate under context_window;
    # prompts are trimmed to leave at least this much room. When context_window
    # is set and max_output_tokens is unset, max_tokens is the window's actual
    # remainder after the prompt (never below this), so long output is not capped
    # at the reservation when the window has headroom. Must be < context_window.
    output_reservation: int = 1024
    # --- Retry policy -------------------------------------------------------
    # Total attempts for one call, including the first. N attempts means N-1
    # sleeps. Per-endpoint because a local server that OOMs under load and a
    # hosted API that rate-limits want different patience.
    max_attempts: int = 3
    # Sleep before each retry, in seconds. The last value repeats if there are
    # more retries than entries, so a single-element list is a flat delay.
    retry_backoff_seconds: list[float] = [5.0, 15.0]

    @field_validator("api_key")
    @classmethod
    def _resolve_env_ref(cls, value: str) -> str:
        return _resolve_env_api_key(value)

    @field_validator("reasoning_effort")
    @classmethod
    def _known_reasoning_effort(cls, value: str | None) -> str | None:
        return _validate_reasoning_effort(value)

    @field_validator("max_attempts")
    @classmethod
    def _at_least_one_attempt(cls, value: int) -> int:
        """0 attempts would never call the endpoint at all."""
        if value < 1:
            raise ValueError(f"max_attempts must be >= 1, got {value}")
        return value

    @field_validator("retry_backoff_seconds")
    @classmethod
    def _non_negative_backoff(cls, value: list[float]) -> list[float]:
        """An empty list is legal (retry immediately); a negative sleep is a typo."""
        for delay in value:
            if delay < 0:
                raise ValueError(
                    f"retry_backoff_seconds entries must be >= 0, got {delay}"
                )
        return value

    @model_validator(mode="after")
    def _reservation_fits_window(self) -> "EndpointConfig":
        # A reservation as wide as the window leaves no room for the prompt: the
        # pruner would shed everything and still overflow. Fail at boot with the
        # exact numbers rather than degrade silently at generation time.
        if (
            self.context_window is not None
            and self.output_reservation >= self.context_window
        ):
            raise ValueError(
                f"output_reservation ({self.output_reservation}) must be smaller "
                f"than context_window ({self.context_window}); otherwise no prompt "
                f"tokens fit"
            )
        # Setting max_output_tokens skips the client's room calculation, so
        # output_reservation becomes the *only* space the prompt trimmer keeps
        # free. A cap wider than the reservation can therefore overrun the
        # window on a prompt trimmed to the floor, and the server truncates the
        # reply — the finish_reason='length' the cap was set to prevent.
        if (
            self.context_window is not None
            and self.max_output_tokens is not None
            and self.max_output_tokens > self.output_reservation
        ):
            raise ValueError(
                f"max_output_tokens ({self.max_output_tokens}) must be <= "
                f"output_reservation ({self.output_reservation}) when "
                f"context_window is set; the prompt is only trimmed to leave "
                f"output_reservation free"
            )
        return self


class AgentEndpointOverride(BaseModel):
    """Sparse per-agent override of the shared endpoint.

    Every field is optional: only the fields set here differ for that agent,
    and everything left unset is inherited from the top-level ``endpoint``. A
    config with no ``agents:`` section behaves exactly as before.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    tokenizer_family: Literal["tiktoken", "char_heuristic"] | None = None
    request_timeout: int | None = None
    stream_read_timeout: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    stream_usage: bool | None = None
    context_window: int | None = None
    output_reservation: int | None = None
    max_attempts: int | None = None
    retry_backoff_seconds: list[float] | None = None

    @field_validator("api_key")
    @classmethod
    def _resolve_env_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _resolve_env_api_key(value)

    @field_validator("reasoning_effort")
    @classmethod
    def _known_reasoning_effort(cls, value: str | None) -> str | None:
        return _validate_reasoning_effort(value)


class GenerationConfig(BaseModel):
    """Generation shape: targets and caps for the draft loop."""

    model_config = ConfigDict(extra="forbid")

    # Whole-manuscript stop signal. 0, empty, or absent means no word limit: the
    # run ends only when every outlined beat is committed. A seeded project's own
    # target takes precedence over this value.
    word_count_target: int | None = None
    revision_retry_cap: int
    max_agent_iterations: int
    recent_prose_beats: int
    # Soft ceiling on the assembled drafting context. Over it, the context node
    # drops the lowest-priority material until the prompt fits.
    context_token_budget: int
    # Proportion of a beat's sentences that may be passive before the audit node
    # faults the draft. A proportion, not a count: 0.25 means a quarter.
    passive_voice_threshold: float
    # How many times the critic is re-prompted when its reply will not parse as
    # failure objects. Distinct from `revision_retry_cap`, which counts
    # draft->audit->revise cycles: this counts "say that again, correctly".
    critic_parse_retries: int
    # Consecutive beats whose critic output stayed unreadable after every retry
    # before the run degrades: validation loosens and the UI warns. A weak model
    # must not silently disable the continuity gate.
    critic_degrade_threshold: int
    # How many times a planner is re-prompted when its reply will not parse as a
    # JSON array. A planner cannot degrade the way the critic can — there is no
    # honest empty plan — so this budget, then a quote repair, then the run dies.
    planner_parse_retries: int
    # When a reply is cut off with zero output, the tokens the server admits to
    # having processed approximate the window it is actually serving. Below this
    # fraction of the declared `endpoint.context_window`, the gap is called out as
    # a server/config mismatch rather than a prompt that is merely too long. A
    # proportion: 0.9 means the served window must reach 90% of the declared one.
    # Slack, because a server counts the prompt with its own tokenizer and stops a
    # little short of the exact boundary.
    served_window_mismatch_fraction: float
    # --- Repetition guard (audit) ------------------------------------------
    # Similarity at or above which two paragraphs count as "the same" prose.
    # A proportion: 0.9 means 90% of the difflib ratio.
    repetition_threshold: float
    # A duplicated run must reach this many consecutive near-duplicate sentences
    # (or a whole paragraph) before it is faulted. Short deliberate refrains stay
    # under the gate and are never touched.
    repetition_min_run: int
    # Shorter verbatim echoes sit below the paragraph gate. Common language is
    # ignored; only a phrase at or above this many words is faulted.
    repetition_min_phrase_words: int = 6
    # A legitimate critic fix may quote a few words (a name, "the third rung")
    # from its passage. Fifteen or more consecutive borrowed words is duplication,
    # not a fix, so the critic remedy is replaced before revision.
    critic_fix_max_borrowed_words: int = 15
    # Phrases the author has declared may recur verbatim. The beat planner may
    # also register a refrain per beat (`intended_refrain`); both are exempt.
    repetition_allowlist: list[str] = []
    # --- Density-gate floor (audit) -----------------------------------------
    # Below this many sentences, the passive/emotion/tic density gates are
    # skipped entirely: a beat with e.g. 3 sentences can only ever quantize to
    # 0%, 33%, 67%, or 100%, so a proportional threshold is either unreachable
    # or automatic — never a meaningful signal.
    passive_min_sentences: int = 6
    # --- Emotion-tell guard (audit) ----------------------------------------
    # Proportion of a beat's sentences that may name an emotion outright before
    # the audit faults the draft for telling rather than showing.
    emotion_word_threshold: float
    # The named-emotion vocabulary the guard counts. Data, not logic: overridable
    # in config, with a sensible default so a fresh config need not restate it.
    # Matched whole-word and case-insensitively, so inflections must be listed
    # explicitly ("fear" does not match "feared").
    emotion_words: list[str] = [
        # fear
        "fear", "feared", "fearful", "afraid", "terror", "terrified",
        "terrifying", "dread", "dreaded", "panic", "panicked", "horror",
        "horrified", "alarm", "alarmed", "fright", "frightened", "scared",
        "nervous", "nervously", "anxiety", "anxious", "apprehension",
        "uneasy", "unease", "worried", "worry",
        # anger
        "rage", "enraged", "fury", "furious", "anger", "angry", "angrily",
        "irritation", "irritated", "annoyed", "annoyance", "resentment",
        "resentful", "indignation", "indignant", "hatred", "loathing",
        "contempt", "spite", "bitterness", "bitterly",
        # sadness
        "grief", "grieving", "sorrow", "sorrowful", "despair", "despairing",
        "misery", "miserable", "anguish", "anguished", "sadness", "sad",
        "sadly", "melancholy", "heartbreak", "heartbroken", "devastated",
        "devastation", "loneliness", "lonely", "regret", "remorse",
        # shame
        "guilt", "guilty", "shame", "ashamed", "humiliation", "humiliated",
        "embarrassment", "embarrassed", "mortified",
        # joy
        "joy", "joyful", "happiness", "happy", "happily", "elation", "elated",
        "euphoria", "euphoric", "delight", "delighted", "glee", "gleeful",
        "excitement", "excited", "excitedly", "relief", "relieved",
        # other named states
        "confusion", "confused", "disgust", "disgusted", "jealousy",
        "jealous", "envy", "envious", "hopeless", "hopelessness", "pride",
        "proud", "longing", "yearning", "desperation", "desperate",
        "frustration", "frustrated",
    ]
    # --- Style-tic guard (audit) --------------------------------------------
    # Proportion of a beat's sentences that may lean on a stock gesture or an
    # abstract emotional shorthand before the audit faults the draft. Separate
    # from the emotion gate so the two can be tuned independently.
    tic_phrase_threshold: float = 0.15
    # The stock-phrase vocabulary the guard counts. Multi-word phrases are
    # matched whole. Data, not logic: overridable in config, defaulted to the
    # tics observed in generated drafts plus the stock gestures and abstract
    # shorthand that editors and AI-prose studies flag most often.
    tic_phrases: list[str] = [
        # --- observed in this project's own drafts -------------------------
        "trembling", "trembled", "deep breath", "shaky breath",
        "tears welled", "welled with tears", "eyes filled with tears",
        "traced the handwriting", "traced the letters", "traced the words",
        "heavy silence", "silence hung", "the weight of",
        "shared history", "legacy", "closure", "bittersweet",
        # --- stock cardiac / respiratory tells ------------------------------
        "heart pounded", "heart pounding", "heart hammered", "heart raced",
        "heart racing", "heart skipped", "pulse quickened", "breath caught",
        "breath hitched", "caught her breath", "caught his breath",
        "let out a breath", "let out a sigh", "released a breath",
        "exhaled slowly", "breath she didn't know", "breath he didn't know",
        # --- throat / stomach tells -----------------------------------------
        "lump in her throat", "lump in his throat", "throat tightened",
        "throat closed", "stomach lurched", "stomach dropped",
        "stomach churned", "stomach twisted", "knot in her stomach",
        "knot in his stomach",
        # --- stock gestures --------------------------------------------------
        "clenched her jaw", "clenched his jaw", "jaw tightened",
        "clenched her fists", "clenched his fists", "furrowed brow",
        "brow furrowed", "raised an eyebrow", "arched an eyebrow",
        "quirked an eyebrow", "ran a hand through her hair",
        "ran a hand through his hair", "raked a hand through",
        "bit her lip", "bit his lip", "chewed her lip",
        "rolled her eyes", "rolled his eyes", "shoulders slumped",
        "shoulders sagged", "squared her shoulders", "squared his shoulders",
        "swallowed hard", "nodded slowly", "shook her head slowly",
        "shook his head slowly",
        # --- eye-contact and chill tells -------------------------------------
        "eyes widened", "eyes narrowed", "eyes darted", "met her eyes",
        "met his eyes", "held her gaze", "held his gaze", "locked eyes",
        "blood ran cold", "blood turned to ice", "chill ran down",
        "shiver ran down", "shiver down her spine", "shiver down his spine",
        # --- voice tells ------------------------------------------------------
        "barely above a whisper", "voice cracked", "voice broke",
        "voice barely a whisper",
        # --- abstract shorthand and AI-prose markers --------------------------
        "a testament to", "tapestry of", "symphony of", "dance of",
        "echoes of", "a reminder that", "served as a reminder",
        "palpable", "nestled", "bustling", "in that moment",
        "in that instant", "little did", "unbeknownst to",
        "couldn't help but", "couldn't shake the feeling",
        "something shifted", "something had changed", "air thick with",
        "the air was thick", "silence stretched", "time seemed to slow",
        "the world narrowed", "a mix of emotions", "wave of emotion",
    ]
    # --- Intensity arc (beat planner) --------------------------------------
    # How many times the beat planner is re-prompted for a varied emotional arc
    # when its plan comes back nearly all high-arousal. Bounded, then accepted.
    planner_intensity_retries: int
    # Absolute target-arousal above which a beat counts as "hot".
    intensity_hot_threshold: float
    # Fraction of a chapter's beats that may be hot before the plan is re-prompted.
    intensity_flat_fraction: float
    # Shortest chapter the arc check applies to. A chapter of one or two beats
    # has no arc to shape, so flagging it as "flat" would be meaningless.
    intensity_min_beats: int = 3
    # --- PAD quantization ----------------------------------------------------
    # Half-width of the neutral band. An axis reading within ±this of zero
    # carries no directional signal and quantizes to "neu". 0.33 splits each
    # axis into three roughly equal thirds of its [-1.0, 1.0] range. Widening it
    # makes beats read as neutral more often; narrowing it makes them read as
    # committed to a direction. The 3x3x3 grid (and pad_baselines.json) is
    # unaffected either way — only where the boundaries fall.
    pad_band_threshold: float = 0.33
    # --- Quote budgets --------------------------------------------------------
    # How much of the offending prose the audit quotes into a failure. This is
    # what the reviser matches against to locate the span, so too short makes
    # spans ambiguous and too long wastes the revision prompt.
    audit_quote_chars: int = 240
    # Longest reply still treated as a clean critic verdict rather than prose
    # the parser should reject.
    critic_verdict_max_chars: int = 240
    # Budget for the full list of offending sentences a density failure appends
    # to its suggested_fix. Deliberately wider than audit_quote_chars, which
    # sizes a single locatable span: this list is instruction prose covering
    # every offender, and truncating it to one quote's width shows the reviser
    # a fraction of what it has to fix.
    audit_offender_list_chars: int = 1200
    # --- Agent tools --------------------------------------------------------
    # Offers `web_search` to every agent when true. Off by default: agents
    # ground themselves in the story's own canon (seed, outline, threads,
    # committed manuscript); the public web is an explicit research mode, not a
    # default reflex.
    research_mode: bool = False
    # How many times any one tool may be called within a single agent loop.
    # Stops a model from spending its bounded iterations re-running the same
    # search instead of answering.
    tool_call_cap: int = 3
    # Wall-clock bound for one tool execution. The loop turns expiry into a
    # structured tool_timeout result instead of hanging the generation task.
    tool_timeout: float = 30.0

    @field_validator("word_count_target", mode="before")
    @classmethod
    def _optional_target(cls, value: object) -> object:
        """0, None, and "" all mean "no limit"; a negative target is a typo."""
        if value is None or value == 0 or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, int) and value < 0:
            raise ValueError(f"word_count_target must be >= 0 or empty, got {value}")
        return value

    @field_validator(
        "passive_voice_threshold",
        "repetition_threshold",
        "emotion_word_threshold",
        "tic_phrase_threshold",
        "intensity_hot_threshold",
        "intensity_flat_fraction",
        "pad_band_threshold",
    )
    @classmethod
    def _proportion(cls, value: float, info: ValidationInfo) -> float:
        """A threshold outside 0..1 silently disables the gate. Fail at boot."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{info.field_name} is a proportion in 0..1, got {value}"
            )
        return value

    @field_validator(
        "critic_parse_retries", "planner_parse_retries", "planner_intensity_retries"
    )
    @classmethod
    def _non_negative(cls, value: int, info: ValidationInfo) -> int:
        """0 is legal — never re-prompt — but a negative budget is a typo."""
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0, got {value}")
        return value

    @field_validator(
        "critic_degrade_threshold",
        "repetition_min_run",
        "repetition_min_phrase_words",
        "critic_fix_max_borrowed_words",
        "tool_call_cap",
        "intensity_min_beats",
        "audit_quote_chars",
        "audit_offender_list_chars",
        "critic_verdict_max_chars",
        "passive_min_sentences",
        # The agent loop raises AgentLoopError on a non-positive budget, and
        # critics now survives that exception rather than propagating it — so a
        # typo'd 0 would silently degrade every beat instead of failing at boot.
        "max_agent_iterations",
    )
    @classmethod
    def _positive(cls, value: int, info: ValidationInfo) -> int:
        """A threshold of 0 would degrade before the first failure ever happened."""
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value

    @field_validator("tool_timeout")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"tool_timeout must be > 0, got {value}")
        return value


class RevisionConfig(BaseModel):
    """How the revision node locates and splices a rewritten span.

    Span mode rewrites only the prose a critic faulted; full mode regenerates
    the beat. These govern when a span is trusted enough to splice — the
    difference between a surgical fix and a whole-beat rewrite.
    """

    model_config = ConfigDict(extra="forbid")

    # Similarity a critic's quoted span must reach against the draft before it
    # counts as located. Below it the "span" is likelier a paraphrase, and
    # rewriting the wrong sentence is worse than rewriting the beat. Raising it
    # sends more beats to full rewrite; lowering it risks mis-splices.
    fuzzy_threshold: float = 0.8
    # Shortest quoted span worth locating. Fuzzy-matching a 3-character needle
    # against a page of prose is noise.
    min_span_chars: int = 4
    # A span rewrite the model padded with copies of the surrounding prose
    # splices in as duplicated paragraphs. A replacement may not exceed this
    # multiple of the span it replaces...
    span_growth_limit: float = 3.0
    # ...or, for a very short span that may legitimately grow more, the span's
    # length plus this many characters. The larger of the two allowances wins.
    span_growth_slack_chars: int = 400
    # A run of this many words from the replacement found verbatim in the prose
    # around the span means the model echoed its context; the splice is rejected
    # and the beat falls back to a full rewrite.
    echo_min_words: int = 8

    @field_validator("fuzzy_threshold")
    @classmethod
    def _proportion(cls, value: float, info: ValidationInfo) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} is a proportion in 0..1, got {value}")
        return value

    @field_validator(
        "min_span_chars", "span_growth_slack_chars", "echo_min_words"
    )
    @classmethod
    def _positive(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value

    @field_validator("span_growth_limit")
    @classmethod
    def _at_least_one(cls, value: float) -> float:
        """Below 1.0 a replacement could never be longer than what it replaces."""
        if value < 1.0:
            raise ValueError(f"span_growth_limit must be >= 1.0, got {value}")
        return value


class ToolsConfig(BaseModel):
    """Retrieval budgets for the agents' read-only story tools.

    Every one of these trades context spend against how much true story an
    agent can see. They are the difference between a drafter that remembers a
    character's voice and one that invents it again.
    """

    model_config = ConfigDict(extra="forbid")

    # --- get_recent_commits -------------------------------------------------
    # Words of each prior beat's tail shown as `closing_words`. This is what the
    # drafter continues from, so it directly sets seam quality between beats.
    recent_commit_closing_words: int = 40
    # --- get_character_sheet ------------------------------------------------
    # Committed lines sampled to show a character's voice, spread across the
    # manuscript so early and late chapters both contribute.
    character_dialogue_samples: int = 8
    # Longest single sampled line before it is truncated.
    character_sample_chars: int = 200
    # --- search_manuscript --------------------------------------------------
    # Characters of matching prose returned per hit.
    search_snippet_chars: int = 300
    # Score added when the whole query appears verbatim, so one exact phrase hit
    # outranks any pile of scattered term hits.
    search_phrase_bonus: int = 25
    # --- find_repetition ----------------------------------------------------
    # Most repetition matches returned, best first.
    repetition_max_matches: int = 10
    # Characters of each matched passage quoted back.
    repetition_snippet_chars: int = 240
    # At or below this word count the input is treated as a phrase and checked
    # verbatim; above it, by paragraph similarity. A difflib ratio on a five-word
    # phrase is noise.
    repetition_phrase_max_words: int = 12
    # --- check_draft --------------------------------------------------------
    # Most offending sentences quoted back by the draft checker.
    check_draft_max_quotes: int = 5
    # Characters of each quoted sentence.
    check_draft_quote_chars: int = 240

    @field_validator("*")
    @classmethod
    def _positive(cls, value: int, info: ValidationInfo) -> int:
        """Every budget here is a count; zero would disable the tool silently."""
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    endpoint: EndpointConfig
    generation: GenerationConfig
    # Both default to their documented values, so a config predating these
    # blocks keeps working and only the keys you want to change need writing.
    revision: RevisionConfig = RevisionConfig()
    tools: ToolsConfig = ToolsConfig()

    # Sparse per-agent inference overrides, keyed by agent role. An agent with
    # no entry uses ``endpoint`` unchanged, so existing single-endpoint configs
    # keep working without modification.
    agents: dict[str, AgentEndpointOverride] = {}

    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "data/museai.db"
    event_log_path: str = "data/events.jsonl"
    # Where `export_manuscript` writes the committed manuscript, and where the
    # manager salvages a best-seen draft on a dead run. Config keys, like
    # `db_path` above, rather than the `Path("data/output")` /
    # `Path("data/drafts")` literals these replaced — a hardcoded, CWD-relative
    # path is invisible to test isolation: on 2026-07-25 a pytest run wrote its
    # own fixture output into the real `data/output/`, silently replacing a
    # 5,657-word manuscript with 25 words from a test fixture, because nothing
    # about the path said "this is configurable, point it elsewhere for tests."
    output_dir: str = "data/output"
    draft_dir: str = "data/drafts"
    # Dev-only reset route guard. Production deployments should set this false.
    allow_reset: bool = True
    # Seconds the agents' web_search tool waits on a search engine.
    web_search_timeout: int = 10

    @field_validator("agents")
    @classmethod
    def _known_agent_roles(
        cls, value: dict[str, AgentEndpointOverride]
    ) -> dict[str, AgentEndpointOverride]:
        """A typo'd agent role must fail at boot, not silently use the default."""
        unknown = sorted(set(value) - set(AGENT_ROLES))
        if unknown:
            raise ValueError(
                f"unknown agent role(s) in agents: {unknown}; "
                f"known roles: {', '.join(AGENT_ROLES)}"
            )
        return value

    def endpoint_for(self, agent: str) -> EndpointConfig:
        """The inference endpoint one agent role actually uses.

        The shared ``endpoint`` with that agent's sparse overrides applied.
        An agent with no override — or a role this config never mentions —
        gets the shared endpoint itself.
        """
        override = self.agents.get(agent)
        if override is None:
            return self.endpoint
        merged = self.endpoint.model_dump()
        merged.update(
            {
                key: value
                for key, value in override.model_dump().items()
                if value is not None
            }
        )
        return EndpointConfig(**merged)


def persist_project_id(project_id: str, path: str | Path = "config.yaml") -> AppConfig:
    """Rewrite ``project_id`` in the config file and return the reloaded config.

    Edits the raw YAML document rather than dumping a resolved ``AppConfig`` so
    unresolved ``${VAR}`` references (the API key) survive the write.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {cfg_path}"
        )
    raw["project_id"] = project_id
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(cfg_path)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Raises ``ConfigError`` on a missing file, invalid YAML, or any unknown key
    or type mismatch (extra keys are forbidden at every level).
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {cfg_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {cfg_path}"
        )

    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {cfg_path}:\n{exc}") from exc
