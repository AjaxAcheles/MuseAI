"""Module: M04 (LLM Inference Boundary)
Render LLM instructions from Jinja2 XML templates by node name.

Prompt *content* lives entirely in ``prompts/*.xml.j2`` templates, never in
Python — tuning wording requires no code change (Architecture_Map §4, split 2).
Rendering is strict: any expected variable that is missing aborts loudly with a
Jinja ``UndefinedError`` rather than silently substituting a blank, so the
generator can never be handed a half-filled instruction.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from jinja2 import Template

# Templates are named "<node_name>.xml.j2" (Architecture_Map §4 file tree).
_TEMPLATE_SUFFIX = ".xml.j2"

# Canonical node/template names only: a "node_" prefix followed by lowercase
# ASCII words. This is also the path-traversal guard — no dots, slashes,
# backslashes, or uppercase can pass, so a name can never escape prompts/.
_NODE_NAME_PATTERN = re.compile(r"^node_[a-z0-9]+(?:_[a-z0-9]+)*$")


class PromptTemplateError(Exception):
    """Base class for prompt-loading failures."""


class InvalidNodeNameError(PromptTemplateError, ValueError):
    """Raised when a node name is not a safe, canonical template name."""


class PromptTemplateNotFoundError(PromptTemplateError, FileNotFoundError):
    """Raised when no template file exists for a (valid) node name."""


def _default_template_dir() -> Path:
    """The prompts/ directory that holds this loader and its templates."""
    return Path(__file__).resolve().parent


class PromptLoader:
    """Strict Jinja2 renderer for node prompt templates rooted at ``prompts/``."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self.template_dir = (
            Path(template_dir) if template_dir is not None else _default_template_dir()
        )
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=StrictUndefined,  # missing variables fail loudly
            autoescape=False,  # prompts are raw XML for the model, not HTML
            keep_trailing_newline=True,
        )

    def template_name_for_node(self, node_name: str) -> str:
        """Return ``<node_name>.xml.j2`` after validating a safe node name.

        Rejects anything that is not a canonical ``node_*`` identifier, which
        makes path traversal (``..``, ``/``, ``\\``) structurally impossible.
        """
        if not isinstance(node_name, str) or not _NODE_NAME_PATTERN.fullmatch(node_name):
            raise InvalidNodeNameError(
                f"unsafe or non-canonical node name: {node_name!r}"
            )
        return f"{node_name}{_TEMPLATE_SUFFIX}"

    def load(self, node_name: str) -> Template:
        """Load the compiled template for ``node_name``."""
        template_name = self.template_name_for_node(node_name)
        try:
            return self._env.get_template(template_name)
        except TemplateNotFound as exc:
            raise PromptTemplateNotFoundError(
                f"no prompt template {template_name!r} in {self.template_dir}"
            ) from exc

    def render(self, node_name: str, context: dict) -> str:
        """Render ``node_name``'s template with ``context``.

        A missing variable raises Jinja's ``UndefinedError`` (via
        ``StrictUndefined``) — it is never silently blanked.
        """
        return self.load(node_name).render(context)
