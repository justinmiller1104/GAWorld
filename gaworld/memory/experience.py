"""Episode, habit, intention and relationship persistence for agents.

Canonical home: :mod:`gaworld.memory.experience`.
Legacy callers should use the ``experience_store`` shim at the project root.
"""

from __future__ import annotations

import json
import os
from typing import Any

from gaworld.settings import CONFIG
from gaworld.settings.behavior import mcf_realism_enabled


def _memory_dir(cfg: dict[str, Any] | None = None) -> str:
    base = cfg if isinstance(cfg, dict) else CONFIG
    return base.get("memory_dir", "output/memory")


def _episodes_path(agent_id: int | str, cfg: dict[str, Any] | None = None) -> str:
    return os.path.join(_memory_dir(cfg), f"agent_{agent_id}_episodes.jsonl")


def _habits_path(agent_id: int | str, cfg: dict[str, Any] | None = None) -> str:
    return os.path.join(_memory_dir(cfg), f"agent_{agent_id}_habits.json")


def _intentions_path(agent_id: int | str, cfg: dict[str, Any] | None = None) -> str:
    return os.path.join(_memory_dir(cfg), f"agent_{agent_id}_intentions.json")


def _relationships_path(agent_id: int | str, cfg: dict[str, Any] | None = None) -> str:
    return os.path.join(_memory_dir(cfg), f"agent_{agent_id}_relationships.json")


def _env_preferences_path(agent_id: int | str, cfg: dict[str, Any] | None = None) -> str:
    return os.path.join(_memory_dir(cfg), f"agent_{agent_id}_env_preferences.json")


def _load_json_dict(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_json_dict(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_agent_episodes(agent_id: int | str, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    path = _episodes_path(agent_id, cfg=cfg)
    if not os.path.exists(path):
        return []
    episodes = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    episodes.append(item)
    except OSError:
        return []
    return episodes


def append_agent_episode(
    agent_id: int | str, episode: dict[str, Any], cfg: dict[str, Any] | None = None
) -> None:
    if not isinstance(episode, dict):
        return
    path = _episodes_path(agent_id, cfg=cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    except OSError:
        return


# ---------------------------------------------------------------------------
# Machine-readable activity spans: one explicit row per activity segment,
# written on every activity transition and closed at run end. The MCF time-use
# scorer prefers these over tick-snapshot reconstruction (which silently lost
# coverage whenever a tick was missed).
# ---------------------------------------------------------------------------

# Open span per agent: {agent_id: {"day", "start", "activity", "location",
# "purpose"}}. Module-level because spans bridge ticks while the per-tick
# agent dict is fresh each tick; _reset_span_state() clears it between runs.
_OPEN_SPANS: dict[int, dict] = {}


def _reset_span_state() -> None:
    _OPEN_SPANS.clear()


def _write_span_row(agent_id: int, span: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    row = {
        "kind": "span",
        "agent_id": agent_id,
        "day": span["day"],
        "start": span["start"],
        "end": span["end"],
        "activity": span["activity"],
        "location": span.get("location", ""),
        "purpose": span.get("purpose", ""),
    }
    append_agent_episode(agent_id, row, cfg=cfg)


def record_activity_span(
    agent_id: int,
    day: int,
    time_str: str,
    activity: str,
    location: str = "",
    purpose: str = "",
    cfg: dict[str, Any] | None = None,
) -> None:
    """Note the agent's current activity; when it differs from the open span's
    activity, close the open span at this tick and open a new one."""
    # Span rows are an MCF realism artifact; a default run writes none.
    if not mcf_realism_enabled():
        return
    open_span = _OPEN_SPANS.get(agent_id)
    if open_span is not None and open_span["activity"] == activity and open_span["day"] == day:
        return
    if open_span is not None:
        open_span["end"] = time_str
        _write_span_row(agent_id, open_span, cfg=cfg)
    _OPEN_SPANS[agent_id] = {
        "day": day,
        "start": time_str,
        "activity": activity,
        "location": location,
        "purpose": purpose,
    }


def close_agent_span(agent_id: int, day: int, time_str: str, cfg: dict[str, Any] | None = None) -> None:
    """Close the agent's open span at run end (or day end) — the final span
    must not stay implicit."""
    if not mcf_realism_enabled():
        _OPEN_SPANS.pop(agent_id, None)
        return
    open_span = _OPEN_SPANS.pop(agent_id, None)
    if open_span is None:
        return
    open_span["end"] = time_str
    _write_span_row(agent_id, open_span, cfg=cfg)


def close_all_agent_spans(
    agents_by_id: dict[int, Any], day: int, time_str: str, cfg: dict[str, Any] | None = None
) -> None:
    """Close every open activity span so the final segment is an explicit row,
    not an implicit tail the scorer has to guess at."""
    for agent_id in agents_by_id:
        close_agent_span(agent_id, day, time_str, cfg=cfg)


def prune_and_decay_episodes(agent_id: int | str, cfg: dict[str, Any] | None) -> None:
    if not isinstance(cfg, dict):
        cfg = {}
    episodes = load_agent_episodes(agent_id)
    if not episodes:
        return
    max_episodes = int(cfg.get("max_episodes_per_agent", 2000))
    current_day = int(cfg.get("current_day", 0))
    half_life = int(cfg.get("decay_half_life_days", 14))
    day_episodes = []
    older = []
    for ep in episodes:
        created_day = int(ep.get("created_at_day", ep.get("day", 0) or 0))
        age = max(0, current_day - created_day)
        if half_life > 0:
            factor = 0.5 ** (age / half_life)
        else:
            factor = 1.0
        salience = float(ep.get("salience", 0.0))
        ep["decayed_salience"] = max(0.0, min(1.0, salience * factor))
        if created_day == current_day:
            day_episodes.append(ep)
        else:
            older.append(ep)
    if len(episodes) > max_episodes:
        # Keep all current-day episodes even if they exceed max.
        keep_quota = max(0, max_episodes - len(day_episodes))
        older_sorted = sorted(
            older,
            key=lambda x: (
                float(x.get("decayed_salience", 0.0)),
                int(x.get("created_at_day", x.get("day", 0) or 0)),
            ),
            reverse=True,
        )
        kept = day_episodes + older_sorted[:keep_quota]
    else:
        kept = day_episodes + older
    kept.sort(
        key=lambda x: (
            int(x.get("day", 0) or 0),
            str(x.get("time", "")),
            str(x.get("episode_id", "")),
        )
    )
    path = _episodes_path(agent_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_agent_habits(agent_id: int | str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return _load_json_dict(_habits_path(agent_id, cfg=cfg))


def save_agent_habits(agent_id: int | str, habits: Any, cfg: dict[str, Any] | None = None) -> None:
    if not isinstance(habits, dict):
        habits = {}
    _save_json_dict(_habits_path(agent_id, cfg=cfg), habits)


def load_agent_intentions(agent_id: int | str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return _load_json_dict(_intentions_path(agent_id, cfg=cfg))


def save_agent_intentions(agent_id: int | str, intentions: Any, cfg: dict[str, Any] | None = None) -> None:
    if not isinstance(intentions, dict):
        intentions = {}
    _save_json_dict(_intentions_path(agent_id, cfg=cfg), intentions)


def load_agent_relationships(agent_id: int | str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return _load_json_dict(_relationships_path(agent_id, cfg=cfg))


def save_agent_relationships(
    agent_id: int | str, relationships: Any, cfg: dict[str, Any] | None = None
) -> None:
    if not isinstance(relationships, dict):
        relationships = {}
    _save_json_dict(_relationships_path(agent_id, cfg=cfg), relationships)


def load_agent_env_preferences(agent_id: int | str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the agent's learned location-aversion preferences (P4)."""
    return _load_json_dict(_env_preferences_path(agent_id, cfg=cfg))


def save_agent_env_preferences(
    agent_id: int | str, env_preferences: Any, cfg: dict[str, Any] | None = None
) -> None:
    """Persist the agent's learned location-aversion preferences (P4)."""
    if not isinstance(env_preferences, dict):
        env_preferences = {}
    _save_json_dict(_env_preferences_path(agent_id, cfg=cfg), env_preferences)
