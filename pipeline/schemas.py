"""Typed contract shared across the pipeline.

Stages 2-4 return these models via Claude structured outputs
(`client.messages.parse(..., output_format=Model)`), so they double as the JSON
schema sent to the API. Keep them to plain fields and lists of objects:
structured outputs do not support recursive schemas or numeric/length
constraints, so we avoid those here.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Brief(BaseModel):
    """Normalized creative request — the pipeline's entry point (built in code)."""

    idea: str = Field(description="The user's freeform creative idea / prompt.")
    source_excerpt: str = Field(
        default="", description="Text extracted from uploaded document(s), if any."
    )
    genre: str = Field(default="", description="Requested genre, or empty for the model to choose.")
    tone: str = Field(default="", description="Requested tone/mood, or empty.")
    pov: str = Field(default="", description="Point of view, e.g. 'first person', 'third limited'.")
    length: str = Field(default="short", description="Length preset: flash|short|novelette|novella.")
    characters: str = Field(default="", description="Key characters the user named, freeform.")

    def to_prompt_block(self) -> str:
        lines = [f"Idea: {self.idea.strip()}"]
        if self.genre:
            lines.append(f"Genre: {self.genre}")
        if self.tone:
            lines.append(f"Tone: {self.tone}")
        if self.pov:
            lines.append(f"Point of view: {self.pov}")
        if self.characters:
            lines.append(f"Key characters the user wants: {self.characters}")
        lines.append(f"Target length: {self.length}")
        if self.source_excerpt.strip():
            lines.append("\nReference material the user provided:\n" + self.source_excerpt.strip())
        return "\n".join(lines)


class Premise(BaseModel):
    title: str = Field(description="A working title for the story.")
    logline: str = Field(description="One or two sentences capturing the whole story.")
    genre: str = Field(description="The genre this story lands in.")
    tone: str = Field(description="The dominant tone/mood.")
    themes: List[str] = Field(description="2-4 central themes.")
    central_conflict: str = Field(description="The core dramatic conflict driving the story.")
    setting_seed: str = Field(description="A short sketch of where/when the story takes place.")


class Character(BaseModel):
    name: str
    role: str = Field(description="e.g. protagonist, antagonist, foil, mentor.")
    description: str = Field(description="Physical and background sketch.")
    motivation: str = Field(description="What this character wants and why.")
    arc: str = Field(description="How this character changes across the story.")
    voice: str = Field(description="How they speak/think — diction, rhythm, verbal tics.")


class StoryBible(BaseModel):
    title: str
    logline: str
    pov: str = Field(description="Narrative point of view to write in.")
    tense: str = Field(description="Narrative tense, e.g. 'past' or 'present'.")
    tone_guide: str = Field(description="Prose-level guidance: mood, pacing, diction.")
    setting: str = Field(description="The world/setting in concrete detail.")
    characters: List[Character] = Field(description="The full cast with arcs and voices.")


class Scene(BaseModel):
    index: int = Field(description="1-based order of this scene in the story.")
    title: str = Field(description="A short scene/chapter title.")
    pov_character: str = Field(description="Whose perspective this scene follows.")
    goal: str = Field(description="What this scene must accomplish for the story.")
    beats: List[str] = Field(description="Ordered story beats that happen in this scene.")
    setting: str = Field(description="Where/when this scene takes place.")


class Outline(BaseModel):
    scenes: List[Scene] = Field(description="The ordered scenes that make up the story.")


# ---- helpers used when prompting later stages -----------------------------

def bible_to_prompt_block(b: StoryBible) -> str:
    cast = "\n".join(
        f"- {c.name} ({c.role}): wants {c.motivation}. Arc: {c.arc}. Voice: {c.voice}. {c.description}"
        for c in b.characters
    )
    return (
        f"TITLE: {b.title}\n"
        f"LOGLINE: {b.logline}\n"
        f"POV: {b.pov}; TENSE: {b.tense}\n"
        f"TONE & PROSE GUIDE: {b.tone_guide}\n"
        f"SETTING: {b.setting}\n"
        f"CAST:\n{cast}"
    )


def scene_to_prompt_block(s: Scene) -> str:
    beats = "\n".join(f"  - {b}" for b in s.beats)
    return (
        f"SCENE {s.index}: {s.title}\n"
        f"POV character: {s.pov_character}\n"
        f"Setting: {s.setting}\n"
        f"Goal: {s.goal}\n"
        f"Beats to hit (in order):\n{beats}"
    )
