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
- The last episode of a day extends to 24:00 in its own category when it is
  sleep (the last evening observation of sleep implies the resident sleeps on
  through midnight) — except a sleep episode directly following paid_work:
  after a work shift the sleep timestamp ends the recorded day, and the
  remainder is unrecorded (unknown). A non-sleep last episode vouches for at
  most 2*tick_minutes of its category when it is observed late in the day
  (>= 18:00 — the evening after it is unobserved and flagged as an incomplete
  day, not padded into the category); an earlier last observation means the
  log stopped mid-activity, and the recorded activity persists to 24:00.
- 24h accounting basis: a day's minutes span its full calendar day. The
  head of a day (00:00 -> its first episode) is the tail of the previous
  night's sleep: it is attributed to the current day in the previous
  recorded day's last category when that is sleep, and otherwise (first
  observed day, or the previous day did not end in sleep) defaults to sleep
  unless the day's own first episode is already sleep — an evening-only log
  that opens on sleep leaves the pre-midnight span genuinely unknown.
  Without this carry, ~7–8h of overnight sleep per day would vanish between
  day N's 24:00 tail and day N+1's first episode (~07:30).
- Out-of-order timestamps: inversions larger than an hour mean the record
  is unordered — the day is sorted by time and the sorted record is treated
  as contiguous (deterministic, lossless). Smaller inversions are write
  jitter: record order is kept and negative differences clamp to zero.
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
# survey_basis: the bulletin reports weekly averages over all days, so a
# weekday-only simulation is not directly comparable without a warning.
# paid_work is a participants-basis average -> score only over employed
# agents. study is inflated by 6-17-year-olds in the survey sample ->
# excluded from scoring by default.
TARGETS_2024: dict[str, dict[str, object]] = {
    "sleep": {"minutes": 586, "basis": "all_persons", "survey_basis": "weekly_average"},
    "personal_care": {"minutes": 56, "basis": "all_persons", "survey_basis": "weekly_average"},
    "meals": {"minutes": 103, "basis": "all_persons", "survey_basis": "weekly_average"},
    "paid_work": {"minutes": 383, "basis": "participants", "survey_basis": "weekly_average"},
    "housework": {"minutes": 77, "basis": "all_persons", "survey_basis": "weekly_average"},
    "leisure": {"minutes": 153, "basis": "all_persons", "survey_basis": "weekly_average"},
    "socializing": {"minutes": 18, "basis": "all_persons", "survey_basis": "weekly_average"},
    "commuting": {"minutes": 50, "basis": "all_persons", "survey_basis": "weekly_average"},
    "study": {
        "minutes": 107,
        "basis": "child_inflated",
        "comparable_by_default": False,
        "survey_basis": "weekly_average",
    },
}

# 2018 城镇 column (附表1 / 附表3, RESULT-3188 §5) as the robustness vector.
# survey_basis: 2018 reports workday and restday averages separately, so an
# explicit day_types flag is required to pick the matching column.
# The 2018 bulletin reports 个人卫生护理+用餐 as one combined 154-min bucket;
# it is kept under meals with that caveat.
TARGETS_2018: dict[str, dict[str, object]] = {
    "sleep": {"minutes": 556, "basis": "all_persons", "survey_basis": "workday_restday_split"},
    "meals": {
        "minutes": 154,
        "basis": "all_persons",
        "survey_basis": "workday_restday_split",
    },  # combined with personal care
    "paid_work": {"minutes": 457, "basis": "participants", "survey_basis": "workday_restday_split"},
    "housework": {"minutes": 79, "basis": "all_persons", "survey_basis": "workday_restday_split"},
    "leisure": {"minutes": 250, "basis": "all_persons", "survey_basis": "workday_restday_split"},
    "commuting": {"minutes": 44, "basis": "all_persons", "survey_basis": "workday_restday_split"},
    "study": {
        "minutes": 29,
        "basis": "child_inflated",
        "comparable_by_default": False,
        "survey_basis": "workday_restday_split",
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
    per_agent_day: dict[int, dict[int, dict[str, float]]] | None = None
    uncertainty_warnings: list[str] | None = None


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

    Episodes are sorted by time within each agent-day (deterministic handling
    of out-of-order records; overlaps then clamp to zero). Only days with
    episodes appear in the result; each day spans the full 24h calendar day
    (see the module docstring for the head/tail accounting basis).
    """
    by_agent_day: dict[tuple[int, int], list[tuple[int, str | None]]] = defaultdict(list)
    for raw in episodes:
        ep = _coerce_episode(raw)
        category, _ = classify_activity(ep.final_activity)
        by_agent_day[(ep.agent_id, ep.day)].append((_parse_clock(ep.time), category))

    result: dict[int, dict[int, dict[str, int]]] = {}
    # Process each agent's days in ascending order so the overnight head of
    # a day can inherit the previous recorded day's terminal category.
    by_agent: dict[int, list[int]] = defaultdict(list)
    for agent_id, day in by_agent_day:
        by_agent[agent_id].append(day)
    for agent_id in sorted(by_agent):
        prev_last_category: str | None = None
        for day in sorted(by_agent[agent_id]):
            stamps = by_agent_day[(agent_id, day)]
            # Determinism contract: timestamps inverted by more than an hour
            # mean the file is unordered — sort the whole day and treat the
            # sorted record as contiguous. Smaller inversions are write
            # jitter: keep record order and clamp negatives to zero.
            contiguous = False
            if any(b[0] < a[0] for a, b in zip(stamps, stamps[1:])):
                max_inversion = max(a[0] - b[0] for a, b in zip(stamps, stamps[1:]) if b[0] < a[0])
                if max_inversion > 60:
                    stamps = sorted(stamps, key=lambda s: s[0])
                    contiguous = True
            minutes = _empty_minutes()
            head = stamps[0][0]
            if head > 0:
                # Overnight carry: the pre-first-episode span is the tail of
                # the previous night's sleep (or, on a first/partial day,
                # default sleep — unless the log itself opens on sleep, which
                # leaves the pre-midnight span genuinely unknown).
                if prev_last_category == "sleep" or stamps[0][1] != "sleep":
                    minutes["sleep"] += head
                else:
                    minutes[UNKNOWN] += head
            for index in range(len(stamps) - 1):
                start, category = stamps[index]
                next_start, next_category = stamps[index + 1]
                gap = max(next_start - start, 0)
                bucket = category or UNKNOWN
                if contiguous:
                    # Repaired ordering: the sorted record is trusted, gaps
                    # belong to the previous category.
                    minutes[bucket] += gap
                elif next_category == category or gap <= 2 * tick_minutes or next_category in _BOUNDARY_ANCHORED:
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
            elif last_category == "sleep":
                # Evening sleep implies the resident sleeps on through
                # midnight: the tail stays sleep (cross-midnight accounting).
                minutes["sleep"] += tail
            else:
                # A non-sleep last episode vouches for at most 2*tick of its
                # category when the observation is late in the day (the
                # evening after it is unobserved, not leisure); an early last
                # observation means the log simply stopped, and the recorded
                # activity persists to 24:00.
                if last_start >= 18 * 60:
                    minutes[last_category or UNKNOWN] += min(tail, 2 * tick_minutes)
                    minutes[UNKNOWN] += max(tail - 2 * tick_minutes, 0)
                else:
                    minutes[last_category or UNKNOWN] += tail
            prev_last_category = last_category
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


def _normalize_daily(
    daily: Mapping[int, Mapping[int, Mapping[str, float] | list[Mapping[str, object]]]],
    tick_minutes: int = 10,
) -> dict[int, dict[int, dict[str, float]]]:
    """Coerce raw episode lists into per-category minutes (score convenience)."""
    normalized: dict[int, dict[int, dict[str, float]]] = {}
    for agent_id, days in daily.items():
        normalized[agent_id] = {}
        for day, value in days.items():
            if isinstance(value, list):
                computed = compute_daily_minutes(value, tick_minutes).get(agent_id, {}).get(day)
                normalized[agent_id][day] = {k: float(v) for k, v in (computed or _empty_minutes()).items()}
            else:
                normalized[agent_id][day] = {k: float(v) for k, v in value.items()}
    return normalized


def score_against_targets(
    daily: Mapping[int, Mapping[int, Mapping[str, float] | list[Mapping[str, object]]]],
    employed_agents: Iterable[int] | None = None,
    day_types: Mapping[int, Mapping[int, str]] | None = None,
) -> ScoreResult:
    """Compare simulated daily minutes against TARGETS_2024.

    paid_work is scored only over employed agents (participants basis); when
    ``day_types`` maps agent -> day -> "workday"/"restday", paid_work is
    scored only over workdays so a weekday-only sim is never silently diluted
    by rest days (and vice versa: 2024 targets are weekly averages — see
    ``survey_basis`` metadata). study is excluded by default
    (child_inflated basis). ``overall_error`` is the mean absolute error
    across comparable categories; ``leftover_minutes`` is the mean unknown
    time per agent-day — unknown time is reported, never dropped.

    ``uncertainty_warnings`` flags low classified coverage (< 0.85) and
    incomplete days (unattributed minutes); ``per_agent_day`` exposes the
    per-agent per-category minutes so external analyses (e.g. the 24h-budget
    paid_work/leisure coupling) are possible without re-deriving them.
    """
    employed = set(employed_agents or ())
    normalized = _normalize_daily(daily)
    agent_days: list[tuple[int, int, Mapping[str, float]]] = [
        (agent_id, day, minutes)
        for agent_id, days in normalized.items()
        for day, minutes in days.items()
    ]

    def mean_over(pairs: list[float]) -> float:
        return sum(pairs) / len(pairs) if pairs else 0.0

    def is_workday(agent_id: int, day: int) -> bool:
        return day_types is None or day_types.get(agent_id, {}).get(day) == "workday"

    per_category: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        if category == "study":
            continue
        if category == "paid_work":
            values = [
                minutes.get(category, 0)
                for agent_id, day, minutes in agent_days
                if agent_id in employed and is_workday(agent_id, day)
            ]
        else:
            values = [minutes.get(category, 0) for _, _, minutes in agent_days]
        sim = mean_over([float(v) for v in values])
        target = float(TARGETS_2024[category]["minutes"])
        per_category[category] = {
            "sim": sim,
            "target": target,
            "error": abs(sim - target),
            "n_agent_days": float(len(values)),
        }

    if not agent_days:
        return ScoreResult(
            per_category=per_category,
            overall_error=0.0,
            coverage=0.0,
            leftover_minutes=0.0,
            per_agent_day={},
            uncertainty_warnings=[],
        )

    warnings: list[str] = []
    for agent_id, day, minutes in agent_days:
        unattributed = float(minutes.get(UNKNOWN, 0))
        if unattributed > 0:
            warnings.append(
                f"incomplete day: agent {agent_id} day {day} has {unattributed:.0f} "
                "unattributed minutes (gaps or unobserved head/tail)"
            )
    cov = coverage(normalized)
    if cov < 0.85:
        warnings.append(f"low coverage: only {cov:.2f} of inferred minutes are classified (< 0.85)")

    overall_error = mean_over([spec["error"] for spec in per_category.values()])
    leftover = mean_over([float(minutes.get(UNKNOWN, 0)) for _, _, minutes in agent_days])
    return ScoreResult(
        per_category=per_category,
        overall_error=overall_error,
        coverage=cov,
        leftover_minutes=leftover,
        per_agent_day=normalized,
        uncertainty_warnings=warnings,
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
