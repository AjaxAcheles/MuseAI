"""End-to-end pipeline smoke test (hits the real Claude API).

Runs every stage with a tiny brief and asserts the structured outputs parse and
one scene streams. Requires ANTHROPIC_API_KEY. Run from the project root:

    python scripts/smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import configure_logging, settings  # noqa: E402
from pipeline import stages  # noqa: E402
from pipeline.schemas import Brief  # noqa: E402


async def main() -> int:
    configure_logging()
    ready, message = settings.config_ready()
    if not ready:
        print(f"Skipping live smoke test — {message}")
        return 0
    print(f"Providers — {settings.active_providers()}\n")

    brief = Brief(
        idea="A retired cartographer is asked to map a town that does not appear on any record.",
        genre="literary mystery",
        tone="quiet, uncanny",
        length="flash",
    )

    print("→ premise…")
    premise = await stages.make_premise(brief)
    assert premise.title and premise.logline, "premise missing fields"
    print(f"   {premise.title}: {premise.logline}")

    print("→ story bible…")
    bible = await stages.make_bible(brief, premise)
    assert bible.characters, "bible has no characters"
    print(f"   {len(bible.characters)} characters, POV={bible.pov}")

    print("→ outline…")
    outline = await stages.make_outline(bible, scene_count=1, target_words=900)
    assert outline.scenes, "outline has no scenes"
    print(f"   {len(outline.scenes)} scene(s)")

    print("→ drafting first scene (streaming)…")
    chars = 0
    async for chunk in stages.draft_scene(
        bible, outline.scenes[0], "", 900, is_first=True, is_last=True, max_tokens=2000
    ):
        chars += len(chunk)
    assert chars > 200, "scene draft suspiciously short"
    print(f"   streamed {chars} characters")

    print("\nSMOKE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
