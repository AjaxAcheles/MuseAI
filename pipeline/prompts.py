"""Prompt templates for each pipeline stage.

Kept deliberately un-prescriptive — the models write better creative work when
given the goal and constraints rather than a rigid step list. Each system prompt
sets the role; the per-stage user content carries the brief / prior outputs.
"""
from __future__ import annotations

# ---- Stage 2: premise ------------------------------------------------------

PREMISE_SYSTEM = (
    "You are a story architect. From a creative brief, distill a compelling premise: "
    "a working title, a sharp logline, the genre and tone, the central conflict, a few "
    "guiding themes, and a seed of the setting. Honor any genre/tone/length the brief "
    "specifies; where the brief is silent, make confident, interesting choices. Avoid "
    "cliché and generic 'chosen one' framing unless the brief asks for it."
)

def premise_user(brief_block: str) -> str:
    return f"Here is the creative brief:\n\n{brief_block}\n\nDesign the premise."


# ---- Stage 3: story bible --------------------------------------------------

BIBLE_SYSTEM = (
    "You are a development editor building a story bible. Given a brief and a premise, "
    "define the narrative point of view and tense, a concrete setting, a prose/tone guide, "
    "and a cast of characters — each with a clear motivation, an arc, and a distinct voice. "
    "Give characters specificity and contradiction; make their voices differ from one another."
)

def bible_user(brief_block: str, premise_block: str) -> str:
    return (
        f"BRIEF:\n{brief_block}\n\nPREMISE:\n{premise_block}\n\n"
        "Build the story bible. Choose a POV and tense that serve this premise."
    )


# ---- Stage 4: outline ------------------------------------------------------

OUTLINE_SYSTEM = (
    "You are a story architect outlining a complete narrative with a satisfying arc — "
    "setup, rising complication, climax, and resolution. Produce exactly the requested "
    "number of scenes. Each scene must advance plot and character, name its POV character "
    "from the cast, and list concrete beats. Scenes should connect causally, not episodically."
)

def outline_user(bible_block: str, scene_count: int, target_words: int) -> str:
    return (
        f"STORY BIBLE:\n{bible_block}\n\n"
        f"Outline this story in exactly {scene_count} scene(s), pacing the arc across them. "
        f"The finished story should run roughly {target_words} words total, so size each "
        "scene's scope accordingly."
    )


# ---- Stage 5: scene drafting ----------------------------------------------
# The story bible is supplied as a cached system block; this is the per-scene
# user turn (the volatile suffix).

DRAFT_SYSTEM_PREFIX = (
    "You are a novelist drafting one scene of a longer story. Write immersive, "
    "publishable prose in the established point of view, tense, and tone. Show through "
    "action, sensory detail, and dialogue rather than summary. Keep characters' voices "
    "consistent with the bible. Do not include scene headings, beat labels, author notes, "
    "or commentary — output only the prose of the scene itself."
)

def draft_user(scene_block: str, rolling_summary: str, target_words: int, is_first: bool, is_last: bool) -> str:
    parts = []
    if rolling_summary:
        parts.append("STORY SO FAR (for continuity — do not repeat verbatim):\n" + rolling_summary)
    else:
        parts.append("This is the opening scene of the story.")
    parts.append("\nSCENE TO WRITE NOW:\n" + scene_block)
    edge = []
    if is_first:
        edge.append("Open the story with a hook; establish voice and stakes early.")
    if is_last:
        edge.append("This is the final scene — bring the central conflict to a resolution.")
    if edge:
        parts.append("\n" + " ".join(edge))
    parts.append(f"\nAim for roughly {target_words} words. Write the scene.")
    return "\n".join(parts)


# ---- rolling summary (between scenes) -------------------------------------

SUMMARY_SYSTEM = (
    "You compress story text into a tight continuity note for the author: what happened, "
    "what changed for each character, and any facts later scenes must stay consistent with. "
    "Be factual and concise — no more than 150 words. Output only the note."
)

def summary_user(previous_summary: str, new_scene_prose: str) -> str:
    prior = f"Existing continuity note:\n{previous_summary}\n\n" if previous_summary else ""
    return (
        f"{prior}New scene just written:\n{new_scene_prose}\n\n"
        "Produce the updated continuity note covering the story so far."
    )


# ---- Stage 6: revision -----------------------------------------------------

REVISE_SYSTEM = (
    "You are a line editor and continuity checker. Revise the full draft into a polished "
    "final: smooth transitions between scenes, fix continuity slips against the story bible, "
    "tighten prose, and strengthen the opening and ending — while preserving the plot, voice, "
    "and the author's intent. Output the complete revised story as clean prose with scene "
    "breaks marked by a single centered line containing only '* * *'. No commentary."
)

def revise_user(bible_block: str, full_draft: str) -> str:
    return (
        f"STORY BIBLE (for continuity):\n{bible_block}\n\n"
        f"FULL DRAFT:\n{full_draft}\n\nReturn the polished final story."
    )
