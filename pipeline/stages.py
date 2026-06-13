"""Per-stage Claude calls.

Structured stages (premise / bible / outline) use `messages.parse` for typed
output. Drafting and revision stream prose. The story bible is passed to the
drafting stage as a cached system block so every scene reuses the same prefix.
"""
from __future__ import annotations

from typing import AsyncIterator

import anthropic

from config import settings
from pipeline import prompts
from pipeline.schemas import (
    Brief,
    Outline,
    Premise,
    StoryBible,
    Scene,
    bible_to_prompt_block,
    scene_to_prompt_block,
)

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


# ---- structured stages -----------------------------------------------------

async def make_premise(brief: Brief) -> Premise:
    resp = await client().messages.parse(
        model=settings.planning_model,
        max_tokens=settings.planning_max_tokens,
        thinking={"type": "adaptive"},
        system=prompts.PREMISE_SYSTEM,
        messages=[{"role": "user", "content": prompts.premise_user(brief.to_prompt_block())}],
        output_format=Premise,
    )
    return resp.parsed_output


async def make_bible(brief: Brief, premise: Premise) -> StoryBible:
    premise_block = (
        f"Title: {premise.title}\nLogline: {premise.logline}\n"
        f"Genre: {premise.genre}; Tone: {premise.tone}\n"
        f"Themes: {', '.join(premise.themes)}\n"
        f"Central conflict: {premise.central_conflict}\nSetting seed: {premise.setting_seed}"
    )
    resp = await client().messages.parse(
        model=settings.planning_model,
        max_tokens=settings.planning_max_tokens,
        thinking={"type": "adaptive"},
        system=prompts.BIBLE_SYSTEM,
        messages=[
            {"role": "user", "content": prompts.bible_user(brief.to_prompt_block(), premise_block)}
        ],
        output_format=StoryBible,
    )
    return resp.parsed_output


async def make_outline(bible: StoryBible, scene_count: int, target_words: int) -> Outline:
    resp = await client().messages.parse(
        model=settings.planning_model,
        max_tokens=settings.planning_max_tokens,
        thinking={"type": "adaptive"},
        system=prompts.OUTLINE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": prompts.outline_user(
                    bible_to_prompt_block(bible), scene_count, target_words
                ),
            }
        ],
        output_format=Outline,
    )
    return resp.parsed_output


# ---- streaming stages ------------------------------------------------------

def _draft_system(bible: StoryBible) -> list[dict]:
    """System prompt + bible, with the bible marked for prompt caching so every
    scene in the same run reuses the cached prefix."""
    return [
        {"type": "text", "text": prompts.DRAFT_SYSTEM_PREFIX},
        {
            "type": "text",
            "text": "STORY BIBLE:\n" + bible_to_prompt_block(bible),
            "cache_control": {"type": "ephemeral"},
        },
    ]


async def draft_scene(
    bible: StoryBible,
    scene: Scene,
    rolling_summary: str,
    target_words: int,
    is_first: bool,
    is_last: bool,
    max_tokens: int,
) -> AsyncIterator[str]:
    user = prompts.draft_user(
        scene_to_prompt_block(scene), rolling_summary, target_words, is_first, is_last
    )
    async with client().messages.stream(
        model=settings.drafting_model,
        max_tokens=max_tokens,
        system=_draft_system(bible),
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def update_summary(previous_summary: str, new_scene_prose: str) -> str:
    resp = await client().messages.create(
        model=settings.planning_model,
        max_tokens=400,
        system=prompts.SUMMARY_SYSTEM,
        messages=[
            {"role": "user", "content": prompts.summary_user(previous_summary, new_scene_prose)}
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def revise(bible: StoryBible, full_draft: str, max_tokens: int) -> AsyncIterator[str]:
    async with client().messages.stream(
        model=settings.drafting_model,
        max_tokens=max_tokens,
        system=prompts.REVISE_SYSTEM,
        messages=[
            {"role": "user", "content": prompts.revise_user(bible_to_prompt_block(bible), full_draft)}
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text
