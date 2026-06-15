"""StoryGenerator — orchestrates the multi-stage pipeline as an async generator.

It yields (event, data) tuples that the Quart route serializes as SSE. Stage
outputs are persisted to the DB as they complete, so the reader view survives a
refresh and export can read the finished story.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, AsyncIterator, Tuple

from config import length_preset, settings
from pipeline import stages
from pipeline.schemas import Brief
from storage import db

log = logging.getLogger("museai.generator")

Event = Tuple[str, dict[str, Any]]

STAGE_LABELS = {
    "premise": "Shaping the premise",
    "bible": "Building the story bible",
    "outline": "Outlining the scenes",
    "draft": "Writing the scenes",
    "revise": "Polishing the final draft",
}


async def run(pid: str, brief: Brief) -> AsyncIterator[Event]:
    preset = length_preset(brief.length)
    scene_target_words = max(300, preset["words"] // max(1, preset["scenes"]))
    current_stage = "init"
    log.info(
        "generation start (pid=%s) length=%s | %s",
        pid, brief.length, settings.active_providers(),
    )
    try:
        await db.set_status(pid, "generating")

        # Stage 2: premise
        current_stage = "premise"
        log.info("[%s] stage: premise", pid)
        yield ("stage_start", {"stage": "premise", "label": STAGE_LABELS["premise"]})
        premise = await stages.make_premise(brief)
        await db.save_stage(pid, "premise", premise.model_dump())
        yield ("stage_data", {"stage": "premise", "payload": premise.model_dump()})

        # Stage 3: story bible
        current_stage = "bible"
        log.info("[%s] stage: bible", pid)
        yield ("stage_start", {"stage": "bible", "label": STAGE_LABELS["bible"]})
        bible = await stages.make_bible(brief, premise)
        await db.save_stage(pid, "bible", bible.model_dump())
        yield ("stage_data", {"stage": "bible", "payload": bible.model_dump()})

        # Stage 4: outline
        current_stage = "outline"
        log.info("[%s] stage: outline", pid)
        yield ("stage_start", {"stage": "outline", "label": STAGE_LABELS["outline"]})
        outline = await stages.make_outline(bible, preset["scenes"], preset["words"])
        await db.save_stage(pid, "outline", outline.model_dump())
        yield ("stage_data", {"stage": "outline", "payload": outline.model_dump()})

        # Stage 5: draft scenes (sequential for continuity)
        current_stage = "draft"
        yield ("stage_start", {"stage": "draft", "label": STAGE_LABELS["draft"]})
        scenes = sorted(outline.scenes, key=lambda s: s.index)
        total = len(scenes)
        rolling_summary = ""
        scene_texts: list[str] = []
        for i, scene in enumerate(scenes):
            is_first, is_last = i == 0, i == total - 1
            log.info("[%s] stage: draft scene %d/%d — %s", pid, i + 1, total, scene.title)
            yield (
                "scene_start",
                {"index": scene.index, "total": total, "title": scene.title},
            )
            parts: list[str] = []
            async for chunk in stages.draft_scene(
                bible,
                scene,
                rolling_summary,
                scene_target_words,
                is_first,
                is_last,
                preset["max_tokens"],
            ):
                parts.append(chunk)
                yield ("draft_delta", {"index": scene.index, "text": chunk})
            scene_prose = "".join(parts).strip()
            scene_texts.append(scene_prose)
            yield ("scene_done", {"index": scene.index})
            if not is_last:
                rolling_summary = await stages.update_summary(rolling_summary, scene_prose)

        # Stage 6: revision (whole story in one pass)
        current_stage = "revise"
        log.info("[%s] stage: revise", pid)
        yield ("stage_start", {"stage": "revise", "label": STAGE_LABELS["revise"]})
        full_draft = "\n\n* * *\n\n".join(scene_texts)
        revise_tokens = _revise_token_budget(preset["words"])
        final_parts: list[str] = []
        async for chunk in stages.revise(bible, full_draft, revise_tokens):
            final_parts.append(chunk)
            yield ("revise_delta", {"text": chunk})
        final_story = "".join(final_parts).strip() or full_draft

        content_md = f"# {bible.title}\n\n{final_story}\n"
        story_id = await db.save_story(pid, bible.title, content_md)
        await db.set_status(pid, "complete")
        log.info("[%s] generation complete — '%s' (story=%s)", pid, bible.title, story_id)
        yield ("done", {"story_id": story_id, "title": bible.title})

    except Exception as exc:  # surface to the client, mark project failed
        log.exception("[%s] generation failed during %s", pid, current_stage)
        await db.set_status(pid, "error")
        yield (
            "error",
            {
                "stage": current_stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "trace": traceback.format_exc(),
            },
        )


def _revise_token_budget(target_words: int) -> int:
    # ~1.6 tokens/word, plus headroom; clamp to a streaming-safe ceiling.
    return max(8000, min(96000, int(target_words * 1.6) + 4000))
