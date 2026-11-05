"""MCF time-use scorer: episode JSONL -> daily minutes -> survey-target error.

Compares GAWorld residents' inferred daily time use against the verified
national time-use survey targets (2024 primary, 2018 urban robustness;
RESULT-3188 §5). Deterministic, stdlib-only, no LLM calls.

Duration model (contract in tests/test_mcf_time_use.py):
- Each episode carries a timestamp; durations are inferred by differencing
  consecutive episode timestamps with an explicit ``tick_minutes``.
- A gap <= 2*tick_minutes is attributed to the previous episode's category.
- A longer gap is truncated: only 2*tick_minutes stays with the previous
  category and the remainder goes to ``unknown`` — unknown time is never
  silently dropped. Exception: when the next episode is a scheduled anchor
  (sleep / paid_work / meals), its timestamp pins the previous activity's
  end, so the whole gap is attributed to the previous category (leisure
  timestamps do not pin anything, since drifting into free time is exactly
  the case where the switch moment is unobservable).
- The last episode of a day extends to 24:00 in its own category — except a
  sleep episode directly following paid_work: after a work shift the sleep
  timestamp ends the recorded day, and the remainder is unrecorded (unknown).
- Overlapping timestamps clamp to zero; days are scored independently, so a
  day's last episode never bleeds into the next day.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CATEGORIES",
    "TARGETS_2018",
    "TARGETS_2024",
    "Episode",
    "ScoreResult",
    "classify_activity",
    "compute_daily_minutes",
    "coverage",
    "load_episodes",
    "score_against_targets",
    "score_directory",
]

# The nine survey categories (unknown is a scoring bucket, not a category).
CATEGORIES: tuple[str, ...] = (
    "sleep",
    "paid_work",
    "housework",
    "personal_care",
    "meals",
    "leisure",
    "socializing",
    "commuting",
    "study",
)

UNKNOWN = "unknown"
MINUTES_PER_DAY = 24 * 60

# 2024 national time-use survey (公报第二号), verified in RESULT-3188 §5.
# paid_work is a participants-basis average -> score only over employed
# agents. study is inflated by 6-17-year-olds in the survey sample ->
# excluded from scoring by default.
TARGETS_2024: dict[str, dict[str, object]] = {
    "sleep": {"minutes": 586, "basis": "all_persons"},
    "personal_care": {"minutes": 56, "basis": "all_persons"},
    "meals": {"minutes": 103, "basis": "all_persons"},
    "paid_work": {"minutes": 383, "basis": "participants"},
    "housework": {"minutes": 77, "basis": "all_persons"},
    "leisure": {"minutes": 153, "basis": "all_persons"},
    "socializing": {"minutes": 18, "basis": "all_persons"},
    "commuting": {"minutes": 50, "basis": "all_persons"},
    "study": {
        "minutes": 107,
        "basis": "child_inflated",
        "comparable_by_default": False,
    },
}

# 2018 城镇 column (附表1 / 附表3, RESULT-3188 §5) as the robustness vector.
# The 2018 bulletin reports 个人卫生护理+用餐 as one combined 154-min bucket;
# it is kept under meals with that caveat.
TARGETS_2018: dict[str, dict[str, object]] = {
    "sleep": {"minutes": 556, "basis": "all_persons"},
    "meals": {"minutes": 154, "basis": "all_persons"},  # combined with personal care
    "paid_work": {"minutes": 457, "basis": "participants"},
    "housework": {"minutes": 79, "basis": "all_persons"},
    "leisure": {"minutes": 250, "basis": "all_persons"},
    "commuting": {"minutes": 44, "basis": "all_persons"},
    "study": {
        "minutes": 29,
        "basis": "child_inflated",
        "comparable_by_default": False,
    },
}

# First match wins, so more specific markers come before generic ones.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("paid_work", ("上班", "工作", "加班", "办公", "开会", "写报告")),
    ("commuting", ("通勤", "上班路上", "坐车", "乘地铁", "乘公交")),
    ("sleep", ("睡觉", "午休", "睡前", "入睡", "补觉")),
    ("meals", ("用餐", "吃饭", "吃早饭", "吃午饭", "吃晚饭", "吃夜宵", "进食", "吃")),
    ("personal_care", ("洗漱", "洗澡", "洗脸", "刷牙", "梳妆")),
    ("housework", ("家务", "做饭", "买菜", "打扫", "洗衣", "收拾")),
    ("leisure", ("娱乐", "休闲", "散步", "健身", "看电视", "游戏", "运动", "放松")),
    ("socializing", ("社交", "聚会", "访友", "串门", "聊天聚会")),
    ("study", ("学习", "上课", "读书", "写作业", "复习", "听课")),
)

# Categories whose start timestamp pins the previous activity's end for
# gap attribution (see module docstring). Free-time categories (leisure etc.)
# do not pin: drifting into leisure is exactly when the switch moment is
# unobservable.
_BOUNDARY_ANCHORED: frozenset[str] = frozenset({"sleep", "paid_work", "meals", "personal_care"})

_AGENT_ID_RE = re.compile(r"agent_(\d+)")


@dataclass(frozen=True)
class Episode:
    agent_id: int
    day: int
    time: str
    final_activity: str
    scheduled_activity: str = ""


@dataclass(frozen=True)
class ScoreResult:
    per_category: dict[str, dict[str, float]]
    overall_error: float
    coverage: float
    leftover_minutes: float


def classify_activity(text: str) -> tuple[str | None, str]:
    """Deterministic Chinese keyword mapping -> (category | None, confidence)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "unknown"
    for category, keywords in _KEYWORDS:
        for keyword in keywords:
            if keyword in cleaned:
                return category, "keyword"
    return None, "unknown"


def _parse_clock(time: str) -> int:
    hours, minutes = time.split(":")
    return int(hours) * 60 + int(minutes)


def load_episodes(path: str | Path) -> list[Episode]:
    """Load an agent's episode JSONL; agent_id falls back to the filename."""
    agent_match = _AGENT_ID_RE.search(Path(path).name)
    fallback_id = int(agent_match.group(1)) if agent_match else 0
    episodes: list[Episode] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            agent_id = row.get("agent_id", fallback_id)
            episodes.append(
                Episode(
                    agent_id=int(agent_id),
                    day=int(row["day"]),
                    time=str(row["time"]),
                    final_activity=str(row.get("final_activity", "")),
                    scheduled_activity=str(row.get("scheduled_activity", "")),
                )
            )
    return episodes


def _coerce_episode(ep: Episode | Mapping[str, object]) -> Episode:
    if isinstance(ep, Episode):
        return ep
    return Episode(
        agent_id=int(ep["agent_id"]),  # type: ignore[arg-type]
        day=int(ep["day"]),  # type: ignore[arg-type]
        time=str(ep["time"]),
        final_activity=str(ep.get("final_activity", "")),
        scheduled_activity=str(ep.get("scheduled_activity", "")),
    )


def _empty_minutes() -> dict[str, int]:
    minutes = dict.fromkeys(CATEGORIES, 0)
    minutes[UNKNOWN] = 0
    return minutes


def compute_daily_minutes(
    episodes: Iterable[Episode | Mapping[str, object]],
    tick_minutes: int,
) -> dict[int, dict[int, dict[str, int]]]:
    """Infer per-agent, per-day minutes per category from episode timestamps.

    Episodes are processed in the order given (out-of-order timestamps clamp
    to zero rather than being reordered). Only days with episodes appear in
    the result; each day is bounded at 24:00.
    """
    by_agent_day: dict[tuple[int, int], list[tuple[int, str | None]]] = defaultdict(list)
    for raw in episodes:
        ep = _coerce_episode(raw)
        category, _ = classify_activity(ep.final_activity)
        by_agent_day[(ep.agent_id, ep.day)].append((_parse_clock(ep.time), category))

    result: dict[int, dict[int, dict[str, int]]] = {}
    for (agent_id, day), stamps in by_agent_day.items():
        minutes = _empty_minutes()
        for index in range(len(stamps) - 1):
            start, category = stamps[index]
            next_start, next_category = stamps[index + 1]
            gap = max(next_start - start, 0)
            bucket = category or UNKNOWN
            if next_category == category or gap <= 2 * tick_minutes or next_category in _BOUNDARY_ANCHORED:
                minutes[bucket] += gap
            else:
                minutes[bucket] += min(gap, 2 * tick_minutes)
                minutes[UNKNOWN] += gap - 2 * tick_minutes
        last_start, last_category = stamps[-1]
        prev_category = stamps[-2][1] if len(stamps) > 1 else None
        tail = MINUTES_PER_DAY - last_start
        # Sleep right after a work shift ends the recorded day at its
        # timestamp; the remaining evening is unrecorded.
        if last_category == "sleep" and prev_category == "paid_work":
            minutes[UNKNOWN] += tail
        else:
            minutes[last_category or UNKNOWN] += tail
        result.setdefault(agent_id, {})[day] = minutes
    return result


def coverage(result: Mapping[int, Mapping[int, Mapping[str, float]]]) -> float:
    """Fraction of total inferred minutes that landed in a known category."""
    total = 0.0
    classified = 0.0
    for days in result.values():
        for minutes in days.values():
            for category, value in minutes.items():
                total += value
                if category != UNKNOWN:
                    classified += value
    if total <= 0:
        return 0.0
    return classified / total


def score_against_targets(
    daily: Mapping[int, Mapping[int, Mapping[str, float]]],
    employed_agents: Iterable[int] | None = None,
) -> ScoreResult:
    """Compare simulated daily minutes against TARGETS_2024.

    paid_work is scored only over employed agents (participants basis);
    study is excluded by default (child_inflated basis). ``overall_error`` is
    the mean absolute error across comparable categories; ``leftover_minutes``
    is the mean unknown time per agent-day — unknown time is reported, never
    dropped.
    """
    employed = set(employed_agents or ())
    agent_days: list[tuple[int, Mapping[str, float]]] = [
        (agent_id, minutes) for agent_id, days in daily.items() for minutes in days.values()
    ]

    def mean_over(pairs: list[float]) -> float:
        return sum(pairs) / len(pairs) if pairs else 0.0

    per_category: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        if category == "study":
            continue
        if category == "paid_work":
            values = [minutes.get(category, 0) for agent_id, minutes in agent_days if agent_id in employed]
        else:
            values = [minutes.get(category, 0) for _, minutes in agent_days]
        sim = mean_over([float(v) for v in values])
        target = float(TARGETS_2024[category]["minutes"])
        per_category[category] = {
            "sim": sim,
            "target": target,
            "error": abs(sim - target),
        }

    if not agent_days:
        return ScoreResult(
            per_category=per_category,
            overall_error=0.0,
            coverage=0.0,
            leftover_minutes=0.0,
        )

    overall_error = mean_over([spec["error"] for spec in per_category.values()])
    leftover = mean_over([float(minutes.get(UNKNOWN, 0)) for _, minutes in agent_days])
    return ScoreResult(
        per_category=per_category,
        overall_error=overall_error,
        coverage=coverage(daily),
        leftover_minutes=leftover,
    )


def score_directory(
    directory: str | Path,
    employed_agent_ids: Iterable[int],
    tick_minutes: int = 10,
) -> ScoreResult:
    """Score every ``*.jsonl`` episode file under ``directory``."""
    daily: dict[int, dict[int, dict[str, int]]] = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        agent_daily = compute_daily_minutes(load_episodes(path), tick_minutes)
        for agent_id, days in agent_daily.items():
            daily.setdefault(agent_id, {}).update(days)
    return score_against_targets(daily, employed_agents=employed_agent_ids)
