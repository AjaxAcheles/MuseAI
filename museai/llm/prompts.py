"""Prompt rendering for MuseAI v1.

Templates live in ``museai/prompts/`` as ``<name>.xml.j2`` and render to plain
XML-structured text — never to HTML — so autoescaping is off. Escaping prose
would corrupt the very thing we are asking the model to continue.

**The message convention.** A template emits a document whose top-level
``<system>`` and ``<user>`` sections become chat messages, in document order::

    <system>
    You are a planner.
    </system>
    <user>
    Plan this arc.
    </user>

:func:`render` gives back the raw rendered text; :func:`render_messages` splits
it into the ``[{"role": ..., "content": ...}]` list that
:func:`museai.llm.client.call_llm` takes.

Undefined template variables are fatal (``StrictUndefined``), matching the
config layer: a mistyped context key fails loudly rather than rendering a blank.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

TEMPLATE_SUFFIX = ".xml.j2"

# The only roles a template may declare as a top-level section.
_ROLES = ("system", "user")

# Matches a top-level <system>…</system> or <user>…</user> block. Non-greedy, so
# the first matching close tag ends the section.
_SECTION_RE = re.compile(
    r"<(?P<role>system|user)>(?P<body>.*?)</(?P=role)>",
    re.DOTALL,
)


class PromptError(RuntimeError):
    """Raised when a template renders to something that is not a valid prompt."""


_env = Environment(
    loader=FileSystemLoader(str(PROMPT_DIR)),
    autoescape=False,  # plain text / XML for a model, not HTML for a browser
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render(template_name: str, /, **context: object) -> str:
    """Render ``museai/prompts/<template_name>.xml.j2`` with ``context``."""
    template = _env.get_template(f"{template_name}{TEMPLATE_SUFFIX}")
    return template.render(**context)


def render_messages(template_name: str, /, **context: object) -> list[dict]:
    """Render a template and split it into chat messages.

    Sections are returned in the order they appear. Both a ``<system>`` and a
    ``<user>`` section are required — a prompt missing either is a template bug,
    not something to paper over at runtime.
    """
    text = render(template_name, **context)

    messages = [
        {"role": match.group("role"), "content": match.group("body").strip()}
        for match in _SECTION_RE.finditer(text)
    ]

    found = {message["role"] for message in messages}
    missing = [role for role in _ROLES if role not in found]
    if missing:
        raise PromptError(
            f"template {template_name!r} rendered no "
            f"{', '.join(f'<{role}>' for role in missing)} section"
        )

    empty = [m["role"] for m in messages if not m["content"]]
    if empty:
        raise PromptError(
            f"template {template_name!r} rendered an empty "
            f"{', '.join(f'<{role}>' for role in empty)} section"
        )

    return messages
