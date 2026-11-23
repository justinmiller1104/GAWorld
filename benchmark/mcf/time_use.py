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
  through midnight). The earlier sleep-after-paid_work exception (a work-shift
  tail recorded as unrecorded) was removed in the Phase 0.9 preflight: it
  contradicted the overnight head-carry, which attributes the same span on
  the next day as sleep — the artifact systematically deflated sleep for
  employed agents. A non-sleep last episode vouches for at
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

Change record (Phase 1.7 Track 2, TASK-3439):
- Structured-field classification (``classify_episode``): an in-transit
  ``travel.status`` (departed/in_transit) or the writer's 乘坐…移动 ``action``
  phrase maps to commuting; 「前往X」 labels that persist past arrival
  (status arrived/stationary) are treated as stale and stay unknown; 「处理待办」
  stays unknown unless a keyword on ``action``/``scheduled_activity`` fires.
  Weak ``scheduled_activity`` keyword hits carry confidence
  "scheduled_keyword". No sleep is ever inferred and gaps are never
  stretched — the duration rules below are unchanged.
- Contamination segmentation (``split_run_segments``): episode files are
  per-agent and were appended across runs, producing duplicate ticks with
  contradictory content. A backward clock jump of more than an hour starts a
  new run segment; segments are returned separately, never silently merged.

Change record (Phase 0.9 preflight, BRIEF-MCF-phase09-green):
- Sleep-after-work artifact removed: a sleep episode directly following
  paid_work no longer ends the recorded day; the evening tail is sleep like
  any other evening (consistent with the overnight head-carry).
- Keyword precision: the bare ``吃`` keyword is applied only when it does
  not start a non-meal compound (吃亏, 吃惊, ...).
- Zero-employed participants: with no employed agent-days, paid_work is
  reported unscored (sim/error = None, n_agent_days = 0) plus an uncertainty
  warning — never a silent 383-minute error.
- Exploratory basis: scoring paid_work with an explicit day_types map marks
  the entry ``basis="exploratory"`` (workday subset vs weekly-average target).
- ``ages_6_plus`` caveat added to the child-inflated study target.
- Tick-mismatch guard: a splittable gap (> 2*tick_minutes) that is not a
  multiple of tick_minutes raises ValueError instead of silently inflating
  ``unknown`` — the file was emitted at a different tick.

Change record (Phase 1.0 GREEN, BRIEF-MCF-phase10-green / TASK-3220):
- Sleep-tail corroboration: a last sleep record evidences "sleeping on
  through midnight" only when it is an evening one (>= 18:00) or a >= 10:00
  record outside a work shift; an early-morning (<= 09:00) or post-work
  (previous episode paid_work) sleep record vouches for at most 2*tick and
  the unobserved tail becomes ``unknown``. The Phase 0.9 expectation that a
  lone 09:00 sleep record credits 900 min to midnight (sleep sim == 1380)
  was over-attribution and is corrected (locked test updated to 500).
- First-day head honesty: a first recorded day defaults its pre-midnight
  head to sleep only when the day ends in a corroborated sleep tail (the
  log spans the night); otherwise the head is genuinely ``unknown``.
- Empty comparison: scoring zero agent-days now warns "no agent-days
  scored" instead of returning a silent ``overall_error == 0.0``; the
  mean error skips categories that are reported unscored (error None).
- style_fit_amplitude range pinned: ``ADMISSIBLE_STYLE_FIT = (0.10, 0.40)``
  with ``validate_style_fit()``; the Phase 0.9 "high = 0.9" is separated as
  ``STRESS_STYLE_FIT`` — a stress point, never an admissible amplitude.
- Survey-comparison honesty gates: ``MIN_COMPARISON_DAYS = 7`` and
  ``MIN_COMPARISON_POPULATION = 20``; ``ScoreResult.frontier_valid`` is
  False (with explanatory warnings) for sub-threshold comparisons — they
  stay scored but are marked exploratory, never like-for-like.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

__all__ = [
    "ADMISSIBLE_STYLE_FIT",
    "CATEGORIES",
    "MIN_COMPARISON_DAYS",
    "MIN_COMPARISON_POPULATION",
    "STRESS_STYLE_FIT",
    "TARGETS_2018",
    "TARGETS_2024",
    "ActivityRecord",
    "Episode",
    "ObservedBreakdown",
    "ScoreResult",
    "classify_activity",
    "classify_activity_record",
    "classify_episode",
    "compute_daily_minutes",
    "compute_observed_minutes",
    "coverage",
    "coverage_breakdown",
    "dedupe_activity_records",
    "load_episodes",
    "score_against_targets",
    "score_directory",
    "split_run_segments",
    "survey_comparison_categories",
    "travel_category",
    "validate_style_fit",
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
TARGETS_2024: dict[str, dict[str, Any]] = {
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
        "ages_6_plus": True,  # the inflated scope covers 6-year-olds and up
    },
}

# 2018 城镇 column (附表1 / 附表3, RESULT-3188 §5) as the robustness vector.
# survey_basis: 2018 reports workday and restday averages separately, so an
# explicit day_types flag is required to pick the matching column.
# The 2018 bulletin reports 个人卫生护理+用餐 as one combined 154-min bucket;
# it is kept under meals with that caveat.
TARGETS_2018: dict[str, dict[str, Any]] = {
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
    ("paid_work", ("上班", "工作", "加班", "办公", "开会", "写报告", "兼职", "接单", "值班", "实习")),
    ("commuting", ("通勤", "上班路上", "坐车", "乘地铁", "乘公交")),
    ("sleep", ("睡觉", "午休", "睡前", "入睡", "补觉")),
    ("meals", ("用餐", "吃饭", "吃早饭", "吃午饭", "吃晚饭", "吃夜宵", "进食")),
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

# Bare ``吃`` is a meal marker only when not starting a non-meal compound
# (吃亏 / 吃惊 / 吃苦 / 吃力 / ...). Specific meal keywords are matched first
# in _KEYWORDS, so this is the last-resort precision fallback.
_BARE_CHI_NON_MEAL = "亏惊苦力劲香醋货透紧不开"

_AGENT_ID_RE = re.compile(r"agent_(\d+)")

# style_fit_amplitude admissible band (Phase 1.0): personality may explain at
# most 10–40% of behavioural variance (gaworld/settings/personality.py
# docstring; scripts/big5_effect_ceiling.py operational ceiling). 0.9 is a
# STRESS value only — the Phase 0.9 spec's "high = 0.9" was never an admissible
# manipulation point (personality explaining >~15% of variance is louder than
# the literature supports).
ADMISSIBLE_STYLE_FIT: tuple[float, float] = (0.10, 0.40)
STRESS_STYLE_FIT: tuple[float, ...] = (0.9,)

# Survey-comparison honesty gates (Phase 1.0): the 2024/2018 targets are
# weekly averages, so a comparison needs >= 7 consecutive simulated days and
# a cohort of >= 20 residents before its overall_error is frontier-valid.
# Sub-threshold comparisons are still scored, but flagged exploratory
# (frontier_valid=False + warnings), never silently reported as like-for-like.
# ~553 agent-days give ±10-min precision on a 120-min-SD category mean; the
# 20×7 floor (~140 agent-days) is the minimum, ~80 agents is survey-grade.
MIN_COMPARISON_DAYS = 7
MIN_COMPARISON_POPULATION = 20


def validate_style_fit(amplitude: float) -> None:
    """Raise ValueError unless ``amplitude`` is inside ADMISSIBLE_STYLE_FIT.

    Stress values (STRESS_STYLE_FIT) are deliberately rejected: they are for
    break-point probes, not admissible manipulation points.
    """
    lo, hi = ADMISSIBLE_STYLE_FIT
    if not lo <= amplitude <= hi:
        raise ValueError(
            f"style_fit amplitude {amplitude} outside the admissible band "
            f"[{lo}, {hi}]; stress values {STRESS_STYLE_FIT} are not admissible "
            "manipulation points"
        )


@dataclass(frozen=True)
class Episode:
    agent_id: int
    day: int
    time: str
    final_activity: str
    scheduled_activity: str = ""
    # Structured activity evidence (Phase 1.7 Track 2). All optional with
    # defaults so old callers and plain dicts keep working unchanged.
    action: str = ""
    location: str = ""
    target_location: str = ""
    travel: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ScoreResult:
    per_category: dict[str, dict[str, object]]
    overall_error: float
    coverage: float
    leftover_minutes: float
    per_agent_day: dict[int, dict[int, dict[str, float]]] | None = None
    uncertainty_warnings: list[str] | None = None
    # False when the duration/population gates (MIN_COMPARISON_DAYS /
    # MIN_COMPARISON_POPULATION) are not met: the comparison is exploratory,
    # not a like-for-like check against weekly-average survey targets.
    frontier_valid: bool = True


def classify_activity(text: str) -> tuple[str | None, str]:
    """Deterministic Chinese keyword mapping -> (category | None, confidence)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "unknown"
    for category, keywords in _KEYWORDS:
        for keyword in keywords:
            if keyword in cleaned:
                return category, "keyword"
    for index, char in enumerate(cleaned):
        if char == "吃" and (index + 1 >= len(cleaned) or cleaned[index + 1] not in _BARE_CHI_NON_MEAL):
            return "meals", "keyword"
    return None, "unknown"


# An episode in transit (``travel.status`` departed/in_transit, or the
# writer's own 乘坐…移动 action phrase) is commuting — the one activity the
# log states machine-readably. Statuses arrived/stationary mean the agent is
# already at the destination: the 「前往X」 label is stale there and the dwell
# purpose is unlogged, so those ticks fall through to the weaker signals and
# usually stay unknown. Purpose-unknown stays unknown: a POI-shaped location
# or 「处理待办」 alone is never guessed into a category.
_IN_TRANSIT_STATUSES = frozenset({"departed", "in_transit"})
_TRAVEL_ACTION_RE = re.compile(r"乘坐.+移动")


def _soft_coerce(ep: Episode | Mapping[str, Any]) -> Episode:
    """Coerce a possibly-partial dict (classification-only) into an Episode."""
    if isinstance(ep, Episode):
        return ep
    travel = ep.get("travel")
    return Episode(
        agent_id=int(ep.get("agent_id", 0)),
        day=int(ep.get("day", 1)),
        time=str(ep.get("time", "00:00")),
        final_activity=str(ep.get("final_activity", "")),
        scheduled_activity=str(ep.get("scheduled_activity", "")),
        action=str(ep.get("action", "")),
        location=str(ep.get("location", "")),
        target_location=str(ep.get("target_location", "")),
        travel=travel if isinstance(travel, Mapping) else None,
    )


def classify_episode(ep: Episode | Mapping[str, Any]) -> tuple[str | None, str]:
    """Classify one episode using structured fields, conservatively.

    Precedence (first match wins, deterministic):
    1. ``travel.status`` in {departed, in_transit} -> commuting (machine-
       readable, definitive).
    2. ``final_activity`` keyword (the Phase 0.7 keyword table).
    3. ``action`` matching 乘坐…移动 -> commuting (the writer's own travel
       phrase; covers tick snapshots whose ``travel`` block was omitted).
    4. ``scheduled_activity`` keyword — weak evidence (a schedule slot the
       agent may have deviated from), reported with its own confidence so a
       consumer can discount it.
    5. unknown. 「处理待办」 and stale 「前往X」 labels are never promoted to a
       category; sleep is never inferred (the log shows none).
    """
    episode = _soft_coerce(ep) if isinstance(ep, Mapping) else ep
    travel = episode.travel or {}
    status = str(travel.get("status", ""))
    # Unified rule (Phase 2.0 §2B): travel status / the writer's travel
    # phrase prove TRAVEL, never a work purpose. Commuting requires the
    # structured destination_purpose (nested in the travel dict, the real
    # logger's placement) — same rule as classify_activity_record.
    if status in _IN_TRANSIT_STATUSES:
        purpose = str(travel.get("destination_purpose", ""))
        if purpose in _COMMUTING_PURPOSES and _VERIFIED_PURPOSE_RE.match(purpose):
            return "commuting", "destination_purpose"
        return travel_category(), "travel_status"
    category, confidence = classify_activity(episode.final_activity)
    if category is not None:
        return category, confidence
    if _TRAVEL_ACTION_RE.search(episode.action or ""):
        return travel_category(), "action_keyword"
    category, _ = classify_activity(episode.scheduled_activity)
    if category is not None:
        return category, "scheduled_keyword"
    return None, "unknown"


def split_run_segments(episodes: Iterable[Episode | Mapping[str, Any]]) -> list[list[Episode]]:
    """Split an episode stream into per-run segments (contamination guard).

    Episode files are per-agent and appended across runs; when a second run
    restarts the clock the file holds duplicate ticks with contradictory
    content. Deterministic detection: a backward jump of more than an hour
    in ``time`` (or a decreasing ``day``) starts a new segment. Never merges
    segments silently — callers score segments separately and choose one.
    """
    segments: list[list[Episode]] = []
    current: list[Episode] = []
    prev_day: int | None = None
    prev_minute: int | None = None
    for raw in episodes:
        ep = _soft_coerce(raw)
        minute = _parse_clock(ep.time)
        reset = current and (
            ep.day < prev_day  # type: ignore[operator]
            or (ep.day == prev_day and prev_minute - minute > 60)  # type: ignore[operator]
        )
        if reset:
            segments.append(current)
            current = []
        current.append(ep)
        prev_day = ep.day
        prev_minute = minute
    if current:
        segments.append(current)
    return segments


def _parse_clock(time: str) -> int:
    hours, minutes = time.split(":")
    return int(hours) * 60 + int(minutes)


def _row_agent_id(path: str | Path, row: Mapping[str, Any]) -> int | None:
    """Resolve a record's agent id; None when it cannot be attributed.

    The filename ``agent_<N>.jsonl`` names the agent; otherwise the row must
    carry a real (non-null) ``agent_id``. Records without either are not
    episodes and must never be bucketed as a phantom agent 0.
    """
    agent_match = _AGENT_ID_RE.search(Path(path).name)
    if agent_match:
        return int(agent_match.group(1))
    agent_id = row.get("agent_id")
    if agent_id is None or (isinstance(agent_id, str) and not agent_id.strip()):
        return None
    return int(agent_id)


def count_unattributed(path: str | Path) -> int:
    """Count JSONL records that cannot be attributed to an agent."""
    unattributed = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if _row_agent_id(path, json.loads(line)) is None:
                unattributed += 1
    return unattributed


def load_episodes(path: str | Path) -> list[Episode]:
    """Load an agent's episode JSONL; agent_id falls back to the filename.

    Records that cannot be attributed to an agent (no ``agent_<N>`` filename
    match and no real ``agent_id`` in the row) are skipped — never bucketed
    as a fake agent; ``score_directory`` reports them via warnings.
    """
    episodes: list[Episode] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Explicit activity spans (sim-side emission) are consumed via
            # load_spans; they are not tick snapshots.
            if str(row.get("kind", "")) == "span":
                continue
            agent_id = _row_agent_id(path, row)
            if agent_id is None:
                continue
            episodes.append(
                Episode(
                    agent_id=int(agent_id),
                    day=int(row["day"]),
                    time=str(row["time"]),
                    final_activity=str(row.get("final_activity", "")),
                    scheduled_activity=str(row.get("scheduled_activity", "")),
                    action=str(row.get("action", "")),
                    location=str(row.get("location", "")),
                    target_location=str(row.get("target_location", "")),
                    travel=row.get("travel") if isinstance(row.get("travel"), Mapping) else None,
                )
            )
    return episodes


def load_spans(path: str | Path) -> list[dict[str, str | int]]:
    """Load the sim-emitted activity spans (``kind == "span"`` rows) from an
    episode JSONL. Empty when the file has none — the caller then falls back
    to tick-snapshot reconstruction."""
    spans: list[dict[str, str | int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("kind", "")) != "span":
                continue
            agent_id = _row_agent_id(path, row)
            if agent_id is None:
                continue
            spans.append(
                {
                    "agent_id": int(agent_id),
                    "day": int(row["day"]),
                    "start": str(row["start"]),
                    "end": str(row["end"]),
                    "activity": str(row.get("activity", "")),
                    "location": str(row.get("location", "")),
                    "purpose": str(row.get("purpose", "")),
                }
            )
    return spans


def spans_daily_minutes(
    spans: Iterable[Mapping[str, Any]],
) -> dict[int, dict[int, dict[str, int]]]:
    """Per-agent, per-day minutes per category straight from span rows —
    no tick-grid reconstruction, so a missed tick cannot lose coverage."""
    result: dict[int, dict[int, dict[str, int]]] = {}
    for span in spans:
        agent_id = int(span["agent_id"])
        day = int(span["day"])
        try:
            start = _parse_clock(str(span["start"]))
            end = _parse_clock(str(span["end"]))
        except ValueError:
            continue
        start = max(0, min(1440, start))
        end = max(0, min(1440, end))
        minutes = max(0, end - start)
        if minutes == 0:
            continue
        purpose = str(span.get("purpose", ""))
        if purpose in _COMMUTING_PURPOSES:
            category = "commuting"
        else:
            cat, _ = classify_activity(str(span.get("activity", "")))
            category = cat or UNKNOWN
        day_map = result.setdefault(agent_id, {}).setdefault(day, _empty_minutes())
        day_map[category] += minutes
    return result


def _coerce_episode(ep: Episode | Mapping[str, Any]) -> Episode:
    if isinstance(ep, Episode):
        return ep
    travel = ep.get("travel")
    return Episode(
        agent_id=int(ep["agent_id"]),
        day=int(ep["day"]),
        time=str(ep["time"]),
        final_activity=str(ep.get("final_activity", "")),
        scheduled_activity=str(ep.get("scheduled_activity", "")),
        action=str(ep.get("action", "")),
        location=str(ep.get("location", "")),
        target_location=str(ep.get("target_location", "")),
        travel=travel if isinstance(travel, Mapping) else None,
    )


def _empty_minutes() -> dict[str, int]:
    minutes = dict.fromkeys(CATEGORIES, 0)
    minutes[UNKNOWN] = 0
    minutes[travel_category()] = 0
    return minutes


def compute_daily_minutes(
    episodes: Iterable[Episode | Mapping[str, Any]],
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
        ep = _soft_coerce(raw)
        category, _ = classify_episode(ep)
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
            if any(b[0] < a[0] for a, b in itertools.pairwise(stamps)):
                max_inversion = max(a[0] - b[0] for a, b in itertools.pairwise(stamps) if b[0] < a[0])
                if max_inversion > 60:
                    stamps = sorted(stamps, key=lambda s: s[0])
                    contiguous = True
            # Sleep-tail corroboration (Phase 1.0): the last sleep record only
            # evidences "sleeping on through midnight" when it is an evening
            # one (>= 18:00), or a >= 10:00 record outside a work shift. An
            # early-morning record (<= 09:00 — possibly still the night, but
            # indistinguishable from a nap/glitch) or a mid-day sleep directly
            # after paid_work (a work-break nap) vouches for at most 2*tick;
            # the rest of the tail is unobserved and must land in ``unknown``,
            # never silently in sleep.
            last_start, last_category = stamps[-1]
            prev_category = stamps[-2][1] if len(stamps) > 1 else None
            if last_category == "sleep":
                if last_start >= 18 * 60:
                    sleep_tail_full = True
                elif last_start < 10 * 60:
                    sleep_tail_full = False
                else:
                    sleep_tail_full = prev_category != "paid_work"
            else:
                sleep_tail_full = False
            minutes = _empty_minutes()
            head = stamps[0][0]
            if head > 0:
                # Overnight carry: the pre-first-episode span is the tail of
                # the previous night's sleep when the previous recorded day
                # ended in sleep. On a FIRST recorded day the head defaults
                # to sleep only when the day itself ends in a corroborated
                # sleep tail (evidence the log spans the night); otherwise —
                # including a log that opens on sleep, or one whose last
                # record is an uncorroborated nap — the head is genuinely
                # unknown.
                if prev_last_category == "sleep":
                    minutes["sleep"] += head
                elif stamps[0][1] == "sleep":
                    minutes[UNKNOWN] += head
                elif prev_last_category is not None or sleep_tail_full:
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
                elif (
                    next_category == category
                    or gap <= 2 * tick_minutes
                    or next_category in _BOUNDARY_ANCHORED
                ):
                    minutes[bucket] += gap
                else:
                    # Tick-mismatch guard: a large gap that isn't a multiple
                    # of tick_minutes means the file was emitted at a
                    # different tick — fail loudly instead of silently
                    # inflating ``unknown``.
                    if gap % tick_minutes:
                        raise ValueError(
                            f"tick mismatch: agent {agent_id} day {day} has a "
                            f"{gap}-minute gap not divisible by tick_minutes={tick_minutes}; "
                            "the episode file was likely emitted at a different tick"
                        )
                    minutes[bucket] += min(gap, 2 * tick_minutes)
                    minutes[UNKNOWN] += gap - 2 * tick_minutes
            tail = MINUTES_PER_DAY - last_start
            if last_category == "sleep" and sleep_tail_full:
                # Corroborated evening sleep implies the resident sleeps on
                # through midnight: the tail stays sleep (cross-midnight
                # accounting; the sleep-after-work exception was removed —
                # see docstring).
                minutes["sleep"] += tail
            elif last_category == "sleep":
                # Uncorroborated sleep record (early-morning or work-break
                # nap): at most 2*tick stays sleep; the unobserved remainder
                # is unknown, never assumed sleep.
                minutes["sleep"] += min(tail, 2 * tick_minutes)
                minutes[UNKNOWN] += max(tail - 2 * tick_minutes, 0)
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
        (agent_id, day, minutes) for agent_id, days in normalized.items() for day, minutes in days.items()
    ]

    def mean_over(pairs: list[float]) -> float:
        return sum(pairs) / len(pairs) if pairs else 0.0

    def is_workday(agent_id: int, day: int) -> bool:
        return day_types is None or day_types.get(agent_id, {}).get(day) == "workday"

    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        if category == "study":
            continue
        if category == "paid_work":
            values = [
                minutes.get(category, 0)
                for agent_id, day, minutes in agent_days
                if agent_id in employed and is_workday(agent_id, day)
            ]
            if not values:
                # Participants basis with zero employed agent-days: a mean
                # over nothing would silently read as a 383-minute error.
                per_category[category] = {
                    "sim": None,
                    "target": float(TARGETS_2024[category]["minutes"]),
                    "error": None,
                    "n_agent_days": 0.0,
                }
                continue
        else:
            values = [minutes.get(category, 0) for _, _, minutes in agent_days]
        sim = mean_over([float(v) for v in values])
        target = float(TARGETS_2024[category]["minutes"])
        spec: dict[str, Any] = {
            "sim": sim,
            "target": target,
            "error": abs(sim - target),
            "n_agent_days": float(len(values)),
        }
        if category == "paid_work" and day_types is not None:
            # Workday-only scoring compares a workday subset against the
            # weekly-average target: exploratory, not a like-for-like check.
            spec["basis"] = "exploratory"
            spec["basis_note"] = "scored on workdays only vs a weekly-average survey target"
        per_category[category] = spec

    if not agent_days:
        return ScoreResult(
            per_category=per_category,
            overall_error=0.0,
            coverage=0.0,
            leftover_minutes=0.0,
            per_agent_day={},
            uncertainty_warnings=[
                "no agent-days scored: an empty comparison cannot be read as a zero-error fit"
            ],
            frontier_valid=False,
        )

    warnings: list[str] = []
    # Survey-comparison honesty gates: the targets are weekly averages, so a
    # sub-threshold duration or cohort is flagged exploratory, not reported
    # as a like-for-like overall_error.
    n_agents = len(normalized)
    n_days = max(len(days) for days in normalized.values())
    frontier_valid = n_days >= MIN_COMPARISON_DAYS and n_agents >= MIN_COMPARISON_POPULATION
    if n_days < MIN_COMPARISON_DAYS:
        warnings.append(
            f"comparison spans only {n_days} day(s) < MIN_COMPARISON_DAYS={MIN_COMPARISON_DAYS}; "
            "weekly-average survey targets need >= 7 simulated days (or an explicit "
            "workday/restday split) — marked exploratory, not frontier-valid"
        )
    if n_agents < MIN_COMPARISON_POPULATION:
        warnings.append(
            f"cohort of {n_agents} agent(s) < MIN_COMPARISON_POPULATION={MIN_COMPARISON_POPULATION}; "
            "a small cohort cannot ground population-level category means — marked "
            "exploratory, not frontier-valid"
        )
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

    overall_error = mean_over([spec["error"] for spec in per_category.values() if spec["error"] is not None])
    leftover = mean_over([float(minutes.get(UNKNOWN, 0)) for _, _, minutes in agent_days])
    if per_category["paid_work"]["n_agent_days"] == 0:
        warnings.append(
            "paid_work: zero employed agent-days scored; the participants-basis "
            "target (383 min) is reported unscored, not as an error"
        )
    return ScoreResult(
        per_category=per_category,
        overall_error=overall_error,
        coverage=cov,
        leftover_minutes=leftover,
        per_agent_day=normalized,
        uncertainty_warnings=warnings,
        frontier_valid=frontier_valid,
    )


def _clamp_minutes(minutes: dict[str, int], budget: int) -> None:
    """Trim ``minutes`` in place to at most ``budget`` total (never below zero)."""
    excess = sum(minutes.values()) - budget
    if excess <= 0:
        return
    take = min(minutes.get(UNKNOWN, 0), excess)
    if take:
        minutes[UNKNOWN] -= take
        excess -= take
    for category in sorted(minutes, key=lambda c: -minutes[c]):
        if excess <= 0:
            break
        take = min(minutes[category], excess)
        minutes[category] -= take
        excess -= take


def score_directory(
    directory: str | Path,
    employed_agent_ids: Iterable[int],
    tick_minutes: int = 10,
) -> ScoreResult:
    """Score the per-agent episode files (``agent_<id>.jsonl``) under ``directory``.

    Only episode files are scored — audit tables such as
    ``intervention_executions.jsonl`` are not episodes. Records that cannot
    be attributed to an agent are reported as an unattributed count in
    ``uncertainty_warnings``, never bucketed as a phantom agent 0.
    """
    daily: dict[int, dict[int, dict[str, int]]] = {}
    unattributed = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        unattributed += count_unattributed(path)
        if not _AGENT_ID_RE.search(path.name):
            continue  # audit tables are not episodes; their unattributed rows are counted above
        # Prefer the sim-emitted activity spans (tick-grid independent):
        # span intervals are authoritative; snapshots outside every span of
        # the same agent-day still count (they fill unspanned gaps). Files
        # without spans fall back to pure snapshot reconstruction.
        spans = load_spans(path)
        if spans:
            span_daily = spans_daily_minutes(spans)
            span_intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for span in spans:
                try:
                    s = max(0, min(1440, _parse_clock(str(span["start"]))))
                    e = max(0, min(1440, _parse_clock(str(span["end"]))))
                except ValueError:
                    continue
                span_intervals.setdefault((int(span["agent_id"]), int(span["day"])), []).append((s, e))
            episodes = [
                ep
                for ep in load_episodes(path)
                if not any(
                    s <= _parse_clock(ep.time) < e for s, e in span_intervals.get((ep.agent_id, ep.day), ())
                )
            ]
            snapshot_daily = compute_daily_minutes(episodes, tick_minutes)
            # Snapshot reconstruction may only attribute the *complement* of
            # the span intervals: compute_daily_minutes sees the remaining
            # boundary snapshots and re-attributes the span-covered minutes
            # between them, which inflates agent-day totals past 1440. Any
            # overflow is the double-counted span overlap — trim ``unknown``
            # first (that overlap is genuinely unattributable), then the
            # reconstructed categories, never the span minutes themselves.
            for agent_id, days in snapshot_daily.items():
                for day, minutes in days.items():
                    span_minutes = sum(span_daily.get(agent_id, {}).get(day, {}).values())
                    _clamp_minutes(minutes, MINUTES_PER_DAY - span_minutes)
            agent_daily: dict[int, dict[int, dict[str, int]]] = {}
            for source in (span_daily, snapshot_daily):
                for agent_id, days in source.items():
                    for day, minutes in days.items():
                        target = agent_daily.setdefault(agent_id, {}).setdefault(day, _empty_minutes())
                        for category, value in minutes.items():
                            target[category] += value
        else:
            agent_daily = compute_daily_minutes(load_episodes(path), tick_minutes)
        for agent_id, days in agent_daily.items():
            daily.setdefault(agent_id, {}).update(days)
    result = score_against_targets(daily, employed_agents=employed_agent_ids)
    if unattributed:
        warnings = list(result.uncertainty_warnings or [])
        warnings.append(
            f"{unattributed} unattributed record(s) dropped: rows without a real agent "
            "identifier cannot be scored"
        )
        result = replace(result, uncertainty_warnings=warnings)
    return result


# ---------------------------------------------------------------------------
# Phase 1.8 Part B.1: observed-only time accounting
# ---------------------------------------------------------------------------


class ObservedBreakdown:
    """Observed-only minutes for one agent-day (Phase 1.8 Contract 1).

    The four observed-only buckets are attributes; the mapping protocol
    (``[]`` / ``in`` / ``get``) exposes them alongside the per-category
    partition so ``breakdown[day]["sleep"]`` and ``breakdown[day]["observed_minutes"]``
    read uniformly. Iteration (``items`` / ``values`` / ``keys``) walks only the
    per-category partition, which sums to the full 24h day: category minutes
    plus ``unknown`` — where ``unknown`` covers BOTH unclassifiable observed
    minutes AND unobserved minutes past the last record (never credited to a
    category). The attributes keep the honest split:

    - ``observed_minutes``: classified + unknown observed (up to the last
      observed record / corroboration, never to midnight on its own).
    - ``classified_minutes``: observed minutes in a known category.
    - ``unknown_minutes``: observed minutes that could not be classified.
    - ``unobserved_minutes``: 1440 - observed (the tail past the last record).
    """

    __slots__ = (
        "classified_minutes",
        "minutes",
        "observed_minutes",
        "unknown_minutes",
        "unobserved_minutes",
    )

    _FIELDS = (
        "observed_minutes",
        "classified_minutes",
        "unknown_minutes",
        "unobserved_minutes",
    )

    def __init__(
        self,
        observed_minutes: int,
        classified_minutes: int,
        unknown_minutes: int,
        unobserved_minutes: int,
        minutes: dict[str, int],
    ):
        self.observed_minutes = observed_minutes
        self.classified_minutes = classified_minutes
        self.unknown_minutes = unknown_minutes
        self.unobserved_minutes = unobserved_minutes
        self.minutes = minutes

    def __getitem__(self, key: str) -> int:
        if key in self._FIELDS:
            return getattr(self, key)
        return self.minutes[key]

    def __contains__(self, key: object) -> bool:
        return key in self._FIELDS or key in self.minutes

    def get(self, key: str, default: int | None = None) -> int | None:
        if key in self._FIELDS:
            return getattr(self, key)
        return self.minutes.get(key, default)

    def keys(self):
        return self.minutes.keys()

    def values(self):
        return self.minutes.values()

    def items(self):
        return self.minutes.items()

    def __iter__(self):
        return iter(self.minutes)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ObservedBreakdown(observed={self.observed_minutes}, "
            f"classified={self.classified_minutes}, unknown={self.unknown_minutes}, "
            f"unobserved={self.unobserved_minutes}, minutes={self.minutes})"
        )


def _sleep_tail_corroborated(last_start: int, last_category: str | None, prev_category: str | None) -> bool:
    """Phase 1.0 corroboration rule (shared with compute_daily_minutes): an
    evening (>= 18:00) sleep record evidences sleeping on through midnight; an
    early-morning (<= 09:00) or work-break (post paid_work) sleep record
    vouches for only a short window."""
    if last_category != "sleep":
        return False
    if last_start >= 18 * 60:
        return True
    if last_start < 10 * 60:
        return False
    return prev_category != "paid_work"


def compute_observed_minutes(
    episodes: Iterable[Episode | Mapping[str, Any]],
    tick_minutes: int,
    window_start: int | None = None,
    window_end: int | None = None,
) -> dict[int, dict[int, ObservedBreakdown]]:
    """Observed-only per-agent, per-day minute accounting (Phase 1.8 Contract 1).

    Unlike :func:`compute_daily_minutes` (which extrapolates a recorded
    activity to midnight when a log simply stops), this scorer never credits
    activity minutes past the last observed record: the tail from the last
    record to 24:00 is unobserved, reported in ``unobserved_minutes`` (and in
    the ``unknown`` bucket of the partition), never in a category. The only
    exception is the Phase 1.0 corroborated evening-sleep tail, where the
    record itself evidences sleep continuing through midnight — that span is
    observed sleep, and it is what lets a cross-midnight sleep span credit
    the next day's head correctly.

    ``window_start`` / ``window_end`` (minutes of day) restrict the observed
    span explicitly; by default the span runs from midnight to the last
    record of the day.
    """
    by_agent_day: dict[tuple[int, int], list[tuple[int, str | None]]] = defaultdict(list)
    for raw in episodes:
        ep = _soft_coerce(raw)
        category, _ = classify_episode(ep)
        by_agent_day[(ep.agent_id, ep.day)].append((_parse_clock(ep.time), category))

    result: dict[int, dict[int, ObservedBreakdown]] = {}
    by_agent: dict[int, list[int]] = defaultdict(list)
    for agent_id, day in by_agent_day:
        by_agent[agent_id].append(day)
    for agent_id in sorted(by_agent):
        prev_corroborated_sleep = False
        for day in sorted(by_agent[agent_id]):
            stamps = sorted(by_agent_day[(agent_id, day)], key=lambda s: s[0])
            minutes = _empty_minutes()
            lo = window_start if window_start is not None else 0
            # Observed span end: an explicit window pins it; otherwise the last
            # record, extended to midnight only by a corroborated sleep tail.
            last_start, last_category = stamps[-1]
            prev_cat = stamps[-2][1] if len(stamps) > 1 else None
            corroborated = _sleep_tail_corroborated(last_start, last_category, prev_cat)
            hi = window_end if window_end is not None else (MINUTES_PER_DAY if corroborated else last_start)
            hi = max(hi, last_start)
            if hi <= lo:
                minutes[UNKNOWN] = MINUTES_PER_DAY
            else:
                # Head before the first in-window record: observed (the sim was
                # running) but unattributed — carried sleep when the previous
                # recorded day ended in corroborated sleep, otherwise unknown.
                first_in_window = next((s for s in stamps if s[0] >= lo and s[0] <= hi), None)
                in_window = [s for s in stamps if lo <= s[0] <= hi]
                if first_in_window is not None and first_in_window[0] > lo:
                    head = first_in_window[0] - lo
                    if prev_corroborated_sleep and window_start is None:
                        minutes["sleep"] += head
                    else:
                        minutes[UNKNOWN] += head
                for index in range(len(in_window) - 1):
                    start, category = in_window[index]
                    gap = in_window[index + 1][0] - start
                    minutes[category or UNKNOWN] += gap
                if corroborated:
                    minutes["sleep"] += MINUTES_PER_DAY - last_start
                elif hi - last_start > 0 and last_start >= lo:
                    # An uncorroborated short tail vouches for at most 2*tick of
                    # the recorded category (the record pins that instant); the
                    # remainder past the last record is unobserved, not unknown-
                    # observed, and stays out of the category partition.
                    minutes[last_category or UNKNOWN] += min(hi - last_start, 2 * tick_minutes)
            partition_total = sum(minutes.values())
            unobserved = max(MINUTES_PER_DAY - partition_total, 0)
            # The reported partition sums to the full day: unobserved minutes
            # fold into ``unknown`` (a scoring bucket, never a category); the
            # ObservedBreakdown fields keep the honest split.
            minutes[UNKNOWN] += unobserved
            classified = sum(v for k, v in minutes.items() if k != UNKNOWN)
            observed = partition_total
            breakdown = ObservedBreakdown(
                observed_minutes=observed,
                classified_minutes=classified,
                unknown_minutes=observed - classified,
                unobserved_minutes=unobserved,
                minutes=dict(minutes),
            )
            result.setdefault(agent_id, {})[day] = breakdown
            prev_corroborated_sleep = corroborated
    return result


def coverage_breakdown(
    episodes: Iterable[Episode | Mapping[str, Any]],
    tick_minutes: int,
) -> dict[str, float]:
    """Observed-span coverage vs full-day coverage, reported separately.

    ``observed_coverage`` = classified / observed minutes; ``full_day_coverage``
    = classified / 1440. The two are kept apart so a half-observed day is not
    diluted into what looks like a poorly classified full day.
    """
    breakdowns = compute_observed_minutes(episodes, tick_minutes)
    classified = 0
    observed = 0
    for days in breakdowns.values():
        for breakdown in days.values():
            classified += breakdown.classified_minutes
            observed += breakdown.observed_minutes
    total_days = sum(len(days) for days in breakdowns.values())
    return {
        "observed_coverage": (classified / observed) if observed > 0 else 0.0,
        "full_day_coverage": classified / (MINUTES_PER_DAY * total_days) if total_days else 0.0,
    }


# ---------------------------------------------------------------------------
# Phase 1.8 Part B.2/B.3: structured activity records, travel vs commuting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityRecord:
    """Machine-readable activity span (Phase 1.8 Contract 2).

    Captured from authoritative tick/schedule state — never from LLM
    inference. ``category`` may be the explicit ``unknown``: nothing is
    fabricated (no invented sleep/work/leisure)."""

    run_id: str
    agent_id: int
    category: str
    start_min: int
    end_min: int
    evidence_source: str


def dedupe_activity_records(records: Iterable[ActivityRecord]) -> list[ActivityRecord]:
    """Deterministically merge identical / overlapping same-category spans.

    Records are sorted by (start, end, category, agent, evidence) and
    same-category spans that overlap or touch are unioned, so overlapping
    evidence is never double-counted regardless of input order.
    """
    ordered = sorted(
        records, key=lambda r: (r.start_min, r.end_min, r.category, r.agent_id, r.evidence_source)
    )
    merged: list[ActivityRecord] = []
    for record in ordered:
        if merged:
            last = merged[-1]
            if (
                last.category == record.category
                and last.agent_id == record.agent_id
                and record.start_min <= last.end_min
            ):
                if record.end_min > last.end_min:
                    merged[-1] = ActivityRecord(
                        run_id=last.run_id,
                        agent_id=last.agent_id,
                        category=last.category,
                        start_min=last.start_min,
                        end_min=record.end_min,
                        evidence_source=last.evidence_source,
                    )
                continue
        merged.append(record)
    return merged


def travel_category() -> str:
    """The first-class category for purpose-unknown journeys (Phase 1.8
    Contract 3): plain travel, distinct from ``commuting``."""
    return "travel"


# Travel vs commuting (Phase 1.8 Contract 3): ``travel.status == in_transit``
# proves travel, never commuting. Commuting requires a verified work purpose
# (a structured ``destination_purpose`` naming a real category); purpose-
# unknown journeys stay travel and are excluded from commuting in survey
# comparisons (see ``survey_comparison_categories``).
_VERIFIED_PURPOSE_RE = re.compile(r"^[a-z_]+$")
# Phase 1.9: only work-related purposes verify commuting — the earlier
# any-snake-case rule promoted even a leisure journey to commuting.
_COMMUTING_PURPOSES = frozenset({"paid_work"})


def classify_activity_record(record: Mapping[str, Any]) -> tuple[str, str]:
    """Classify one structured activity snapshot -> (category, evidence_source).

    Precedence (deterministic):
    1. authoritative structured ``activity_category`` -> ``structured``;
    2. ``travel.status == in_transit`` -> commuting ONLY with a structured
       verified ``destination_purpose``, otherwise plain ``travel``;
    3. keyword inference (weakest evidence, marked ``keyword``);
    4. ``unknown`` — never fabricated sleep/work/leisure.
    """
    structured = record.get("activity_category")
    if structured:
        return str(structured), "structured"
    travel = record.get("travel")
    if isinstance(travel, Mapping) and str(travel.get("status", "")) in _IN_TRANSIT_STATUSES:
        # The real episode logger nests destination_purpose INSIDE the
        # travel dict (0/132 top-level in the Phase 2.0 dry run of
        # record) — read both placements (Phase 2.0 §2C).
        purpose = str(travel.get("destination_purpose") or record.get("destination_purpose") or "")
        if purpose in _COMMUTING_PURPOSES and _VERIFIED_PURPOSE_RE.match(purpose):
            return "commuting", "destination_purpose"
        return travel_category(), "travel_status"
    category, confidence = classify_activity(str(record.get("final_activity", "")))
    if category is not None:
        return category, confidence
    category, _ = classify_activity(str(record.get("scheduled_activity", "")))
    if category is not None:
        return category, "scheduled_keyword"
    return UNKNOWN, "unknown"


def survey_comparison_categories() -> dict[str, list[str]]:
    """Category buckets used when comparing against time-use surveys.

    ``travel`` is a first-class bucket; purpose-unknown journeys are NEVER
    folded into ``commuting`` (``commuting_aliases`` holds only the category
    itself), so unknown-purpose travel cannot inflate the commuting estimate.
    """
    return {
        "travel": [travel_category()],
        "commuting": ["commuting"],
        "commuting_aliases": ["commuting"],
    }
