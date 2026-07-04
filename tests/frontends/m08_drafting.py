"""Module: M06 (Drafting & Generation)
Streamlit frontend for the real drafting node over temp stores.

Drives the production drafting path end to end: context assembly
(``build_context_package``) -> the injected streamed call seam -> the anti-slop
black-box contract -> ``current_draft_text``. The default path is a **scripted
stream** (an editable text area whose content is chunked and fed through the
injected ``call_seam``, with no network at all) plus an **inject failing seam**
checkbox to demonstrate the clean-failure path. An explicit **Live drafter
endpoint** toggle (off by default) instead runs the node with its own real
default seam against the configured ``endpoints.drafter``; on any failure
(missing secrets, unreachable endpoint) the page shows a clear notice and falls
back to the scripted stream rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st
import yaml

from shared import (
    PLANNING_SNAPSHOT_ID,
    ROOT,
    context_defaults,
    display_package_layers,
    output_label,
    page_intro,
    reset_workspace,
    section_header,
    seed_drafting_beat_plan,
    seed_narrative_data,
    seed_planning_data,
    two_section_help,
    workspace,
)
from core.antislop import detect_slop, resolve_slop
from core.config_loader import load_config
from fsm.nodes.node_assemble_context import build_context_package
from fsm.nodes.node_draft_prose import node_draft_prose
from fsm.state import FSM_Pointer
from memory import sqlite_db

_ENV_PATH = ROOT / ".env"

_DEFAULT_SCRIPTED_STREAM = (
    "The bell over the door had gone quiet an hour ago. Elena straightened the "
    "last stack of paperbacks by the window and let herself look at Marcus, who "
    "was still crouched by the shelf he'd built, running a thumb along a seam "
    "that didn't need checking twice. Neither of them said anything. The street "
    "outside had gone the color of dusk, and for once she didn't reach for words "
    "to fill the quiet."
)


def _load_env_file(path: Any) -> list[str]:
    """Load KEY=VALUE lines from ``path`` into ``os.environ`` (existing env wins).

    Minimal dotenv semantics — mirrors ``m07_planning.py``'s helper exactly, kept
    local per the "each frontend imports real code, no shared test-only glue"
    convention. Variables already present in the environment are never overwritten.
    """
    loaded: list[str] = []
    path = Path(path)
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _drafting_config(*, live: bool) -> tuple[Any, dict[str, Any] | None]:
    """Build the config-shaped object for both context assembly and drafting.

    Offline (default): a ``SimpleNamespace`` reading real ``config.yaml`` context/
    runtime values, no secrets required — ``endpoints.drafter`` is never actually
    reached because the page always injects its own ``call_seam``. Live: the real
    validated ``AppConfig`` after loading ``.env`` secrets, so the node's own default
    seam can reach the configured ``endpoints.drafter``.
    """
    if live:
        loaded_keys = _load_env_file(_ENV_PATH)
        config = load_config(ROOT / "config.yaml")
        info = {
            "env_file": str(_ENV_PATH),
            "env_file_found": _ENV_PATH.exists(),
            "env_keys_loaded": loaded_keys,
            "drafter_endpoint": {
                "base_url": config.endpoints.drafter.base_url,
                "model_name": config.endpoints.drafter.model_name,
            },
        }
        return config, info
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    defaults = context_defaults()
    return (
        SimpleNamespace(
            endpoints=SimpleNamespace(
                drafter=SimpleNamespace(
                    tokenizer_family="char_heuristic", model_name="synthetic-drafter"
                )
            ),
            context=SimpleNamespace(**defaults),
            runtime=SimpleNamespace(
                inference_timeout_seconds=raw["runtime"]["inference_timeout_seconds"]
            ),
        ),
        None,
    )


def _chunk_text(text: str) -> list[str]:
    """Split scripted stream text into whitespace-bounded chunks, join-safe.

    Each chunk keeps its trailing whitespace so ``"".join(chunks) == text`` exactly
    — the same round-trip property the Done-check's fake seams rely on.
    """
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch.isspace():
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _make_scripted_seam(chunks: list[str], *, fail: bool, capture: dict[str, Any]):
    """Build a fake call seam yielding ``chunks`` one at a time, capturing what it saw.

    ``capture`` records the rendered prompt (from ``messages``) and the joined raw
    text — the seam's ``messages`` argument is the documented injection point through
    which the page observes the node's real template render without modifying
    ``node_draft_prose``. When ``fail`` is set, the seam emits roughly half the
    chunks then raises, demonstrating the clean mid-stream failure path.
    """

    async def _seam(messages, *, on_token, timeout_seconds):
        capture["messages"] = messages
        collected: list[str] = []
        fail_at = len(chunks) // 2 if fail else None
        for i, chunk in enumerate(chunks):
            await on_token(chunk)
            collected.append(chunk)
            if fail_at is not None and i == fail_at:
                capture["raw_text"] = "".join(collected)
                raise RuntimeError("scripted failing seam: transport exploded mid-stream")
        raw = "".join(collected)
        capture["raw_text"] = raw
        return raw

    return _seam


def _list_beat_nodes(db_path: str, scene_id: str) -> list[dict[str, Any]]:
    """Planned beat PlanningNodes under ``scene_id``, ordered by beat_index."""
    scene_node = f"{PLANNING_SNAPSHOT_ID}:scene:{scene_id}"
    rows: list[dict[str, Any]] = []
    for node in sqlite_db.get_planning_nodes_by_parent(db_path, PLANNING_SNAPSHOT_ID, scene_node):
        if node.get("level") != "beat":
            continue
        try:
            plan = json.loads(node.get("purpose") or "{}")
        except ValueError:
            plan = {}
        rows.append(
            {
                "beat_id": plan.get("beat_id") or node.get("title") or node["node_id"],
                "beat_index": node.get("ordering", 0),
                "immediate_objective": plan.get("immediate_objective")
                or node.get("summary")
                or "",
            }
        )
    rows.sort(key=lambda r: r["beat_index"])
    return rows


def _run_draft(
    *,
    paths: dict[str, Path],
    pointer: FSM_Pointer,
    scripted_text: str,
    fail_seam: bool,
    live: bool,
) -> dict[str, Any]:
    """Assemble the context package, then run the real drafting node.

    Mirrors the production order (``node_assemble_context`` -> ``node_draft_prose``).
    In Live mode the node runs with ``call_seam=None`` — its own real default seam
    against ``config.endpoints.drafter`` — so no page-side re-implementation of the
    call ever substitutes for the production path; a failure there falls back to the
    scripted stream automatically rather than crashing the page.
    """
    db_path = str(paths["db"])
    log_path = str(paths["event_log"])
    capture: dict[str, Any] = {}
    live_error: str | None = None

    config, llm_info = _drafting_config(live=live)
    base_state: dict[str, Any] = {
        "project_id": "proj-1",
        "fsm_pointer": pointer,
        "app_config": config,
        "sqlite_db_path": db_path,
        "planning_snapshot_id": PLANNING_SNAPSHOT_ID,
    }
    package = build_context_package(base_state)
    base_state["active_context_package"] = package

    chunk_feed: list[str] = []

    async def _publisher(chunk: str) -> None:
        chunk_feed.append(chunk)

    ran_live = False
    if live:
        try:
            state = dict(base_state, streaming_buffer="", current_draft_text="")
            result_state = asyncio.run(node_draft_prose(state, publisher=_publisher))
            ran_live = True
        except Exception as exc:  # noqa: BLE001 - fall back to the scripted stream
            live_error = f"{type(exc).__name__}: {exc}"
            chunk_feed.clear()

    if not ran_live:
        chunks = _chunk_text(scripted_text)
        seam = _make_scripted_seam(chunks, fail=fail_seam, capture=capture)
        offline_config, _ = _drafting_config(live=False)
        state = dict(
            base_state,
            app_config=offline_config,
            streaming_buffer="",
            current_draft_text="",
        )
        try:
            result_state = asyncio.run(
                node_draft_prose(state, call_seam=seam, publisher=_publisher)
            )
            draft_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never crashes it
            result_state = state
            draft_error = f"{type(exc).__name__}: {exc}"
    else:
        draft_error = None

    raw_text = capture.get("raw_text")
    if raw_text is None and ran_live:
        # M11 is an honest no-op passthrough, so the resolved text already equals
        # the raw text — there is no separate pre-antislop copy to capture outside
        # the node when it built its own real default seam (Live mode).
        raw_text = result_state.get("current_draft_text") or ""
    findings = detect_slop(raw_text) if raw_text else []
    resolved = resolve_slop(raw_text, findings) if raw_text else ""

    return {
        "package": package,
        "state": result_state,
        "rendered_prompt": capture.get("messages", [{}])[0].get("content")
        if capture.get("messages")
        else None,
        "chunk_feed": chunk_feed,
        "raw_text": raw_text,
        "findings": findings,
        "resolved": resolved,
        "draft_error": draft_error,
        "ran_live": ran_live,
        "live_error": live_error,
        "llm_info": llm_info,
    }


def _render_package_summary(package: dict[str, Any]) -> None:
    section_header(
        "Assembled context package",
        two_section_help(
            "The layered context package handed to the drafter template: canonical facts, epistemic beliefs, summaries, flavour, temporal context, and macro constraints, plus how token budgeting pruned it.",
            "build_context_package(state) — the same function node_assemble_context calls in the wired graph.",
        ),
    )
    meta = package.get("meta", {})
    cols = st.columns(4)
    cols[0].metric("Initial tokens", meta.get("initial_token_total", "—"))
    cols[1].metric("Final tokens", meta.get("final_token_total", "—"))
    cols[2].metric("Over budget", str(meta.get("over_budget", "—")))
    cols[3].metric("Pruned layers", len(meta.get("pruned_layers", []) or []))
    with st.expander("Layer availability (meta)"):
        st.json({k: v for k, v in meta.items() if k not in ("token_sizing",)})
    display_package_layers(package)


def _render_prompt_and_stream(run: dict[str, Any]) -> None:
    section_header(
        "Rendered prompt + live chunk feed",
        two_section_help(
            "The exact prompt node_draft_prose rendered from node_draft_prose.xml.j2, and the stream chunks the injected publisher received, in arrival order.",
            "Rendered prompt is captured from the call seam's `messages` argument (the documented injection point) — only available in Scripted mode; Live mode uses the node's own real default seam.",
        ),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        output_label(
            "Rendered drafter prompt",
            two_section_help(
                "The full rendered XML prompt sent to the model.",
                "Captured from call_seam(messages, ...); None in Live mode.",
            ),
        )
        if run["rendered_prompt"] is not None:
            with st.expander("Show rendered prompt", expanded=False):
                st.code(run["rendered_prompt"], language="xml")
        else:
            st.info("Not captured in Live mode — the node built its own real default seam.")
    with col_b:
        output_label(
            "Live chunk feed (arrival order)",
            two_section_help(
                "Every chunk the injected publisher received, in the order it arrived — this is what a real SSE subscriber would see live.",
                "Appended by the page's publisher callable, injected via node_draft_prose(publisher=...).",
            ),
        )
        if run["chunk_feed"]:
            st.dataframe(
                [{"order": i, "chunk": c} for i, c in enumerate(run["chunk_feed"])],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No chunks were published for this run.")


def _render_antislop_and_result(run: dict[str, Any]) -> None:
    section_header(
        "Anti-slop step + final state",
        two_section_help(
            "The completed text passed through the M11 detect-then-resolve contract before becoming current_draft_text. Today detect_slop always returns no findings and resolve_slop returns its input unchanged — an honest no-op the node calls anyway so it needs no change when M11's internals are replaced.",
            "detect_slop(raw_text) / resolve_slop(raw_text, findings), re-run here for display against the same text the node processed.",
        ),
    )
    state = run["state"]
    col_a, col_b = st.columns(2)
    with col_a:
        output_label(
            "Anti-slop findings",
            two_section_help(
                "Findings detect_slop reported — expected empty today.",
                "detect_slop(raw_text); core/antislop.py's honest no-op stub.",
            ),
        )
        st.json(run["findings"])
        matches = run["resolved"] == run["raw_text"]
        st.metric("resolved == raw", str(matches))
    with col_b:
        output_label(
            "current_draft_text vs streaming_buffer",
            two_section_help(
                "The node's final state contract: the anti-slop-resolved text becomes current_draft_text, and streaming_buffer clears to \"\" once the draft completes (no stream is in flight).",
                "state['current_draft_text'] / state['streaming_buffer'] as returned by node_draft_prose.",
            ),
        )
        st.text_area(
            "current_draft_text", state.get("current_draft_text", ""), height=160, disabled=True
        )
        st.text_input("streaming_buffer", repr(state.get("streaming_buffer", "")), disabled=True)
    if run["draft_error"]:
        st.error(f"Call seam raised: {run['draft_error']}")
        st.caption(
            "current_draft_text is left unchanged (shown above) — the failure path never "
            "writes a half-written draft as the working text."
        )
    if run["live_error"]:
        st.warning(
            f"Live drafter call failed, fell back to the scripted stream: {run['live_error']}",
            icon="⚡",
        )
    if run["llm_info"]:
        with st.expander("Live drafter endpoint info"):
            st.json(run["llm_info"])


def render() -> None:
    paths = workspace()
    page_intro(
        "M08 Drafting & Generation",
        two_section_help(
            "This page runs the real drafting node end to end: context assembly, the template render, a streamed call through an injectable seam, the anti-slop contract, and the final state. The default path is a fully offline scripted stream; an explicit toggle lets you try a real drafter endpoint instead.",
            "Drives build_context_package + node_draft_prose against throwaway temp stores; call_seam/publisher are the documented injection points.",
        ),
    )

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button(
        "Seed stores",
        width="stretch",
        help=two_section_help(
            "Writes the narrative seed, layers the planning surface on top, then adds one planned-but-undrafted beat (scene-1_b3) with a real PAD behavioural-constraint string, so a draft run has a genuine beat plan to render.",
            "seed_narrative_data -> seed_planning_data -> seed_drafting_beat_plan; all real store APIs, idempotent.",
        ),
    ):
        try:
            seed_narrative_data(paths["db"], paths["provisional"])
            seed_planning_data(paths["db"], event_log_path=paths["event_log"])
            seed_drafting_beat_plan(paths["db"])
            st.success("Seeded narrative + planning stores + one drafting beat plan.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
    if action_cols[1].button(
        "Reset temp workspace",
        width="stretch",
        help=two_section_help(
            "Clears the temp stores and the last run. Use this to start over.",
            "shared.reset_workspace(); also drops the stored m08 run.",
        ),
    ):
        st.session_state.pop("m08_run", None)
        reset_workspace()
    with action_cols[2]:
        output_label(
            "Temp workspace",
            two_section_help(
                "This page reads and writes only inside this session-scoped temp directory.",
                "workspace() paths: SQLite DB, event log, provisional store.",
            ),
        )
        st.code(str(paths["root"]))

    section_header(
        "Choose a beat and run controls",
        two_section_help(
            "Pick which seeded beat to draft, edit the scripted stream text, optionally inject a failing seam to see the clean-failure path, or turn on the Live drafter toggle to call a real configured endpoint instead.",
            "FSM_Pointer resolves the beat plan via its PlanningNode; the scripted stream is chunked and fed through the injected call_seam.",
        ),
    )
    pointer_col, run_col = st.columns([1, 2])
    with pointer_col:
        arc_id = st.text_input("arc_id", "arc-1")
        chapter_id = st.text_input("chapter_id", "chapter-1")
        scene_id = st.text_input("scene_id", "scene-1")
        beats = _list_beat_nodes(str(paths["db"]), scene_id)
        if not beats:
            st.info("No planned beat found under this scene yet — press Seed stores.")
            beat_choice = None
        else:
            beat_choice = st.selectbox(
                "beat to draft",
                beats,
                format_func=lambda b: f"{b['beat_id']} (index {b['beat_index']}) — {b['immediate_objective'][:60]}",
                help=two_section_help(
                    "Which planned-but-undrafted beat to run the drafter against.",
                    "Lists beat-level PlanningNodes parented under this scene's planning node.",
                ),
            )
    with run_col:
        scripted_text = st.text_area(
            "scripted stream text (chunked and fed to the fake seam)",
            value=_DEFAULT_SCRIPTED_STREAM,
            height=140,
            help=two_section_help(
                "This text is split into whitespace-bounded chunks and streamed through the injected call seam one at a time, exactly like a real token stream — edit it to try different drafts.",
                "_chunk_text(text); \"\".join(chunks) reconstructs the original text exactly.",
            ),
        )
        fail_seam = st.checkbox(
            "inject failing seam (raises mid-stream)",
            value=False,
            help=two_section_help(
                "When on, the scripted seam emits roughly half its chunks then raises, so you can watch the clean-failure path: the error surfaces and current_draft_text is left unchanged.",
                "Ignored in Live mode — a real endpoint failure demonstrates this path instead.",
            ),
        )
        live = st.toggle(
            "Live drafter endpoint",
            value=False,
            help=two_section_help(
                "Off by default (fully offline). When on, the node runs with its own real default seam against the configured endpoints.drafter, using secrets from .env. Any failure (missing secrets, unreachable endpoint) shows a notice and falls back to the scripted stream — this page never fakes a backend and never crashes.",
                "call_seam=None so node_draft_prose builds its production default; on exception, re-runs with the scripted seam.",
            ),
        )

    if st.button("Run draft", type="primary", disabled=beat_choice is None):
        try:
            pointer = FSM_Pointer(
                arc_id=arc_id,
                chapter_id=chapter_id,
                scene_id=scene_id,
                beat_index=int(beat_choice["beat_index"]),
                beat_id=str(beat_choice["beat_id"]),
            )
            with st.spinner("Running the real drafting path…"):
                st.session_state.m08_run = _run_draft(
                    paths=paths,
                    pointer=pointer,
                    scripted_text=scripted_text,
                    fail_seam=fail_seam,
                    live=live,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")

    run = st.session_state.get("m08_run")
    if not run:
        st.info("Seed the stores, pick a beat, then run a draft.")
        return

    _render_package_summary(run["package"])
    _render_prompt_and_stream(run)
    _render_antislop_and_result(run)


def main() -> None:
    st.set_page_config(page_title="M08 Drafting", layout="wide")
    render()


if __name__ == "__main__":
    main()
