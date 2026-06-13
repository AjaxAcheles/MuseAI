"""Per-stage model calls.

Each stage goes through an `LLMBackend` (see pipeline/llm.py) chosen per stage
group in config, so the pipeline runs against Claude or a local OpenAI-compatible
model. Structured stages (premise / bible / outline) use `parse`; drafting and
revision `stream`; the rolling summary uses `complete`.
"""
from __future__ import annotations

from typing import AsyncIterator, Tuple

from config import settings
from pipeline import prompts
from pipeline.llm import LLMBackend, get_backend
from pipeline.schemas import (
    Brief,
    Outline,
    Premise,
    StoryBible,
    Scene,
    bible_to_prompt_block,
    scene_to_prompt_block,
)


def _planning() -> Tuple[LLMBackend, str]:
    return get_backend(settings.planning_provider), settings.planning_model


def _drafting() -> Tuple[LLMBackend, str]:
    return get_backend(settings.drafting_provider), settings.drafting_model


# ---- structured stages -----------------------------------------------------

async def make_premise(brief: Brief) -> Premise:
    backend, model = _planning()
    return await backend.parse(
        model,
        prompts.PREMISE_SYSTEM,
        prompts.premise_user(brief.to_prompt_block()),
        Premise,
        settings.planning_max_tokens,
    )


async def make_bible(brief: Brief, premise: Premise) -> StoryBible:
    backend, model = _planning()
    premise_block = (
        f"Title: {premise.title}\nLogline: {premise.logline}\n"
        f"Genre: {premise.genre}; Tone: {premise.tone}\n"
        f"Themes: {', '.join(premise.themes)}\n"
        f"Central conflict: {premise.central_conflict}\nSetting seed: {premise.setting_seed}"
    )
    return await backend.parse(
        model,
        prompts.BIBLE_SYSTEM,
        prompts.bible_user(brief.to_prompt_block(), premise_block),
        StoryBible,
        settings.planning_max_tokens,
    )


async def make_outline(bible: StoryBible, scene_count: int, target_words: int) -> Outline:
    backend, model = _planning()
    return await backend.parse(
        model,
        prompts.OUTLINE_SYSTEM,
        prompts.outline_user(bible_to_prompt_block(bible), scene_count, target_words),
        Outline,
        settings.planning_max_tokens,
    )


# ---- streaming stages ------------------------------------------------------

async def draft_scene(
    bible: StoryBible,
    scene: Scene,
    rolling_summary: str,
    target_words: int,
    is_first: bool,
    is_last: bool,
    max_tokens: int,
) -> AsyncIterator[str]:
    backend, model = _drafting()
    # Two system segments: the role, then the (cacheable) story bible.
    segments = [prompts.DRAFT_SYSTEM_PREFIX, "STORY BIBLE:\n" + bible_to_prompt_block(bible)]
    user = prompts.draft_user(
        scene_to_prompt_block(scene), rolling_summary, target_words, is_first, is_last
    )
    async for text in backend.stream(model, segments, user, max_tokens):
        yield text


async def update_summary(previous_summary: str, new_scene_prose: str) -> str:
    backend, model = _planning()
    return await backend.complete(
        model,
        prompts.SUMMARY_SYSTEM,
        prompts.summary_user(previous_summary, new_scene_prose),
        400,
    )


async def revise(bible: StoryBible, full_draft: str, max_tokens: int) -> AsyncIterator[str]:
    backend, model = _drafting()
    user = prompts.revise_user(bible_to_prompt_block(bible), full_draft)
    async for text in backend.stream(model, [prompts.REVISE_SYSTEM], user, max_tokens):
        yield text
