"""Shared token-budget pruning for assembled prompts.

One drop-loop, used by both the drafting context node and the planners: render
the prompt with the endpoint's own tokenizer, and while it exceeds the budget,
shed context cheapest-source-first. Each drop step mutates shared mutable
context (pops from a list) and reports whether it removed anything; an exhausted
step yields to the next. When every step is spent and the prompt still overflows,
the caller is told (``over_budget``) rather than the prompt being silently
mutilated — the protected core (the beat/chapter instructions) is never dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from museai.llm.tokenizer import count_message_tokens


def pop_front(items: list) -> bool:
    """Drop the oldest item; True if one was removed. For prose/history."""
    if items:
        items.pop(0)
        return True
    return False


def pop_back(items: list) -> bool:
    """Drop the last item; True if one was removed. For priority-sorted lists."""
    if items:
        items.pop()
        return True
    return False


def window_budget(endpoint, *, fallback: int | None = None) -> int | None:
    """Prompt-token budget implied by an endpoint's declared context window.

    ``context_window - output_reservation`` when the window is declared — the
    room the prompt may occupy while still leaving the reservation free to
    generate. ``None`` window returns ``fallback`` (planners pass ``None`` to
    skip trimming; drafting passes its ``context_token_budget`` so the tighter
    of the two limits wins).
    """
    if endpoint.context_window is None:
        return fallback
    reservation = endpoint.output_reservation or 0
    # Never negative: a reservation wider than the window would ask the pruner to
    # shed everything and still report over budget. EndpointConfig rejects that
    # combination at boot, but clamp defensively for any direct caller.
    room = max(0, endpoint.context_window - reservation)
    return room if fallback is None else min(fallback, room)


def prune_to_budget(
    *,
    budget: int,
    render: Callable[[], list[dict]],
    tokenizer_family: str,
    model_name: str,
    drops: Sequence[tuple[str, Callable[[], bool]]],
) -> dict[str, Any]:
    """Shed context via ``drops`` (cheapest first) until ``render()`` fits ``budget``.

    Each ``drops`` entry is ``(name, step)``; ``step`` mutates shared context and
    returns whether it removed something. Returns a report with the token counts
    and per-source drop tallies.
    """

    def count() -> int:
        return count_message_tokens(render(), tokenizer_family, model_name)

    before = count()
    dropped: dict[str, int] = {name: 0 for name, _ in drops}
    for name, step in drops:
        while count() > budget and step():
            dropped[name] += 1
    after = count()
    return {
        "budget": budget,
        "tokens_before": before,
        "tokens": after,
        "dropped": dropped,
        "over_budget": after > budget,
    }
