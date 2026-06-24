"""Module: M04 (LLM Inference Boundary)
Synthetic tests for the strict Jinja2 prompt loader.

Templates are created in ``tmp_path`` so no real prompt content is authored
here; the loader is exercised purely against synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import UndefinedError

from prompts.prompt_loader import (
    InvalidNodeNameError,
    PromptLoader,
    PromptTemplateNotFoundError,
)


def _write_template(template_dir: Path, node_name: str, body: str) -> None:
    (template_dir / f"{node_name}.xml.j2").write_text(body, encoding="utf-8")


def test_successful_render_from_temporary_template(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "node_draft_prose",
        "<instructions>Write about {{ topic }} in {{ tone }} tone.</instructions>",
    )
    loader = PromptLoader(template_dir=tmp_path)
    rendered = loader.render(
        "node_draft_prose", {"topic": "the harbor", "tone": "wistful"}
    )
    assert rendered == "<instructions>Write about the harbor in wistful tone.</instructions>"


def test_missing_variable_raises_undefined_error(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "node_plan_beat_pad",
        "<pad>{{ pleasure }}/{{ arousal }}/{{ dominance }}</pad>",
    )
    loader = PromptLoader(template_dir=tmp_path)
    # 'dominance' is absent -> StrictUndefined aborts loudly, never blanks it.
    with pytest.raises(UndefinedError):
        loader.render("node_plan_beat_pad", {"pleasure": 0.4, "arousal": -0.2})


def test_missing_template_raises_clear_exception(tmp_path: Path) -> None:
    loader = PromptLoader(template_dir=tmp_path)
    with pytest.raises(PromptTemplateNotFoundError):
        loader.load("node_craft_consultant")


@pytest.mark.parametrize(
    "bad_name",
    [
        "../secrets",
        "node_draft_prose/../../etc/passwd",
        "..\\windows",
        "node_draft_prose.xml.j2",  # already suffixed -> dots are not allowed
        "Node_Draft",  # uppercase
        "draft_prose",  # missing canonical node_ prefix
        "",
        "node_",
    ],
)
def test_invalid_node_name_is_rejected(tmp_path: Path, bad_name: str) -> None:
    loader = PromptLoader(template_dir=tmp_path)
    with pytest.raises(InvalidNodeNameError):
        loader.template_name_for_node(bad_name)


def test_path_traversal_cannot_escape_template_dir(tmp_path: Path) -> None:
    # A sibling file outside the template dir must be unreachable even though
    # it exists on disk.
    secret = tmp_path.parent / "outside.xml.j2"
    secret.write_text("<secret>leaked</secret>", encoding="utf-8")
    template_root = tmp_path / "prompts"
    template_root.mkdir()
    loader = PromptLoader(template_dir=template_root)
    with pytest.raises(InvalidNodeNameError):
        loader.render("node_x/../../outside", {})


def test_rendered_output_preserves_xml_tags_as_plain_text(tmp_path: Path) -> None:
    # autoescape is off: literal tags stay literal AND an injected value that
    # contains angle brackets is passed through verbatim, not HTML-escaped.
    _write_template(
        tmp_path,
        "node_adversarial_critics_continuity",
        "<critic role=\"continuity\"><draft>{{ draft }}</draft></critic>",
    )
    loader = PromptLoader(template_dir=tmp_path)
    rendered = loader.render(
        "node_adversarial_critics_continuity",
        {"draft": 'He said <em>"go"</em>.'},
    )
    assert rendered == (
        '<critic role="continuity"><draft>He said <em>"go"</em>.</draft></critic>'
    )
    assert "&lt;" not in rendered  # no HTML escaping occurred


def test_default_template_dir_points_at_prompts_package() -> None:
    loader = PromptLoader()
    assert loader.template_dir.name == "prompts"
    assert (loader.template_dir / "prompt_loader.py").exists()
