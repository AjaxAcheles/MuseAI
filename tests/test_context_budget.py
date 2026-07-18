"""The shared prompt-budget pruning helper (museai.fsm.nodes.context_budget)."""

from __future__ import annotations

from museai.core.config import EndpointConfig
from museai.fsm.nodes.context_budget import (
    pop_back,
    pop_front,
    prune_to_budget,
    window_budget,
)
from museai.llm.tokenizer import count_message_tokens


def _endpoint(**kw) -> EndpointConfig:
    return EndpointConfig(
        base_url="https://x.invalid/v1",
        api_key="k",
        model_name="m",
        tokenizer_family="char_heuristic",
        **kw,
    )


class TestWindowBudget:
    def test_no_window_returns_the_fallback(self):
        assert window_budget(_endpoint(), fallback=8000) == 8000

    def test_no_window_and_no_fallback_is_none(self):
        # Planners pass no fallback: None means "do not trim".
        assert window_budget(_endpoint()) is None

    def test_declared_window_leaves_the_reservation_free(self):
        ep = _endpoint(context_window=8192, output_reservation=1024)
        assert window_budget(ep) == 8192 - 1024

    def test_a_small_window_tightens_the_fallback(self):
        ep = _endpoint(context_window=4096, output_reservation=1024)
        assert window_budget(ep, fallback=8000) == 4096 - 1024

    def test_a_generous_window_lets_the_fallback_win(self):
        ep = _endpoint(context_window=100_000, output_reservation=1024)
        assert window_budget(ep, fallback=8000) == 8000


class TestPruneToBudget:
    @staticmethod
    def _render_factory(protected: str, prose: list[str], threads: list[str]):
        def render() -> list[dict]:
            return [{"role": "user", "content": protected + "".join(prose + threads)}]

        return render

    def test_nothing_is_dropped_when_it_already_fits(self):
        prose, threads = ["p"], ["t"]
        render = self._render_factory("core", prose, threads)
        report = prune_to_budget(
            budget=10_000,
            render=render,
            tokenizer_family="char_heuristic",
            model_name="m",
            drops=[("prose", lambda: pop_front(prose)), ("threads", lambda: pop_back(threads))],
        )
        assert report["dropped"] == {"prose": 0, "threads": 0}
        assert report["over_budget"] is False

    def test_prose_is_exhausted_before_any_thread_is_dropped(self):
        prose = ["x" * 400 for _ in range(5)]
        threads = ["y" * 40 for _ in range(2)]
        # Budget = the prompt with all prose gone. The loop stops once it fits, so
        # threads are never reached.
        empty = self._render_factory("core", [], threads)()
        budget = count_message_tokens(empty, "char_heuristic", "m")
        render = self._render_factory("core", prose, threads)
        report = prune_to_budget(
            budget=budget,
            render=render,
            tokenizer_family="char_heuristic",
            model_name="m",
            drops=[("prose", lambda: pop_front(prose)), ("threads", lambda: pop_back(threads))],
        )
        assert report["dropped"]["prose"] > 0
        assert report["dropped"]["threads"] == 0
        assert report["over_budget"] is False
        assert threads == ["y" * 40, "y" * 40]

    def test_over_budget_is_reported_when_the_protected_core_will_not_fit(self):
        prose, threads = ["p" * 100], ["t" * 100]
        render = self._render_factory("core" * 500, prose, threads)
        report = prune_to_budget(
            budget=10,
            render=render,
            tokenizer_family="char_heuristic",
            model_name="m",
            drops=[("prose", lambda: pop_front(prose)), ("threads", lambda: pop_back(threads))],
        )
        # Everything droppable is gone, yet the core alone still overflows.
        assert prose == [] and threads == []
        assert report["over_budget"] is True
