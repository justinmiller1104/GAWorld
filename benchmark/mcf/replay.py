"""LLM replay wrapper (Phase 2.5): identical pre-trigger history via a
canary-style memoizer over ``gaworld.llm.providers.call_llm``.

A checkpoint fork (see ``benchmark.mcf.experiment.fork_arms_from_checkpoint``)
restores identical agent/plugin/RNG state into both arms, but the LLM still
samples stochastically at the first call — replay closes that gap: pre-trigger
calls are served from a cache of recorded responses keyed on the exact request
(task, agent_id, system, temperature, prompt); past the replay window (the
stop predicate, e.g. the intervention trigger tick) calls fall through to the
live provider and fresh sampling resumes.

Installation mirrors the proven canary seam (``benchmark.mcf.launch``):
module-attribute rebind of ``providers.call_llm``, restorable via the
returned ``restore()``. A pre-stop cache MISS raises :class:`ReplayCacheMiss`
loudly — a silent fall-through would silently diverge the shared history the
whole protocol rests on.

No network, no inference of its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import gaworld.llm.providers as providers

__all__ = [
    "ReplayCacheMiss",
    "ReplayMismatch",
    "install_llm_replay",
    "install_sequence_replay",
    "sim_time_stop_predicate",
]

_CLOCK_RE = re.compile(r"第\s*(\d+)\s*天.*?现在是\s*(\d{1,2}):(\d{2})")


def _parse_clock_stamp(prompt: Any) -> tuple[int, int] | None:
    """(day, minutes-since-midnight) from the sim's clock line stamped into
    perception prompts (``第 {day} 天 ... 现在是 {time}``), else None."""
    if not isinstance(prompt, str):
        return None
    match = _CLOCK_RE.search(prompt)
    if match is None:
        return None
    day = int(match.group(1))
    minutes = int(match.group(2)) * 60 + int(match.group(3))
    return day, minutes


def sim_time_stop_predicate(day: int, time: str) -> StopPredicate:
    """Authoritative sim-time stop boundary: fires on the first call whose
    prompt's clock line is at or after ``day``/``time`` (the intervention
    trigger tick) — never before it, never after it.

    Call it as ``sim_time_stop_predicate(day=1, time="12:00")``; the returned
    predicate is installed as the replay's ``stop_predicate``.
    """
    hour, minute = time.split(":")
    boundary = (int(day), int(hour) * 60 + int(minute))

    def predicate(prompt, *args, **kwargs) -> bool:
        stamp = _parse_clock_stamp(prompt)
        return stamp is not None and stamp >= boundary

    return predicate


class ReplayMismatch(Exception):
    """A replayed call does not match the recorded sequence.

    Raised for a request whose task/agent_id/system/temperature/prompt differs
    from the recorded event at the current position, and for unexpected calls
    (extra / duplicated / beyond the recording) while the window is open — a
    silent fall-through would silently diverge the shared history. The message
    names the exact event index (and field, for field mismatches).
    """


class SequenceReplay:
    """Ordered replay over recorded events; installed over providers.call_llm.

    Unlike the Phase 2.5 dict memoizer (which loses repeated identical
    prompts — 1,558 calls / 1,099 unique keys), every call replays its OWN
    recorded response, in recording order. Once the stop predicate fires the
    window is closed for good (latched): every later call goes to the live
    base llm and consumes no event.
    """

    def __init__(
        self,
        recorded_events: list[dict[str, Any]],
        base_llm: Callable[..., Any],
        stop_predicate: StopPredicate | None,
    ):
        self._events = [dict(event) for event in recorded_events]
        self._base = base_llm
        self._stop = stop_predicate
        self._cursor = 0
        self._hits = 0
        self._stopped = False
        # a never-recorded prompt is tolerated only when an explicit live base
        # was supplied (Phase 2.6: a partial window over a live run, calls
        # shape-identified); with the offline fallback (the fork protocol) the
        # replay IS the protocol and every call must match its recording
        self._live_base = not isinstance(base_llm, _OfflineFallback)
        self._latch_prompt: Any = None
        self._tolerated = False
        self._first_mismatch: dict[str, Any] | None = None

    def __call__(self, prompt, *args, **kwargs):
        if self._stopped:
            return self._after_stop(prompt, *args, **kwargs)
        if self._stop is not None and self._stop(prompt, *args, **kwargs):
            # latching: past the boundary the window never re-opens, and a
            # live fall-through consumes no recorded event — unless the window
            # is closing over never-replayed events that the prompt-unverified
            # history cannot vouch were post-trigger ("missing")
            self._stopped = True
            self._latch_prompt = prompt
            remaining = self._events[self._cursor :]
            if (
                self._cursor > 0
                and remaining
                and not self._tolerated
                and not any(event.get("prompt") == prompt for event in self._events)
            ):
                mismatch = {
                    "index": self._cursor,
                    "kind": "missing",
                    "expected": remaining[0].get("prompt"),
                    "got": prompt,
                }
                if self._first_mismatch is None:
                    self._first_mismatch = mismatch
                raise ReplayMismatch(
                    f"missing call at replay index {self._cursor}: window closed at {prompt!r} with "
                    f"{len(remaining)} unconsumed event(s), first expected {remaining[0].get('prompt')!r}"
                )
            return self._base(prompt, *args, **kwargs)
        index = self._cursor
        got = (
            kwargs.get("task"),
            kwargs.get("agent_id"),
            kwargs.get("system"),
            kwargs.get("temperature"),
            prompt,
        )
        if index >= len(self._events):
            mismatch = {"index": index, "kind": "unexpected", "expected": None, "got": got}
            if self._first_mismatch is None:
                self._first_mismatch = mismatch
            raise ReplayMismatch(
                f"unexpected call at replay index {index}: no recorded event remains "
                f"(got task={got[0]!r}, agent_id={got[1]!r}, prompt={got[4]!r})"
            )
        event = self._events[index]
        incoming_prompt = got[4]
        if incoming_prompt != event.get("prompt"):
            # any prompt that is not the current event's is a sequence fault:
            # a LATER event's prompt is reordered; an already-consumed event's
            # is a duplicated dispatch; a prompt matching NO recorded event was
            # never recorded and would silently diverge the shared history.
            # The one tolerance kept from Phase 2.6 (see _live_base above) also
            # marks the history prompt-unverified: no "missing" claim later.
            later = any(e.get("prompt") == incoming_prompt for e in self._events[index + 1 :])
            consumed = any(e.get("prompt") == incoming_prompt for e in self._events[:index])
            if later:
                kind = "reordered"
            elif consumed:
                kind = "duplicate"
            elif self._live_base:
                kind = None
            else:
                kind = "unexpected"
            if kind is not None:
                mismatch = {
                    "index": index,
                    "kind": kind,
                    "field": "prompt",
                    "expected": event.get("prompt"),
                    "got": incoming_prompt,
                }
                if self._first_mismatch is None:
                    self._first_mismatch = mismatch
                raise ReplayMismatch(
                    f"{kind} call at replay index {index}: field 'prompt' expected {event.get('prompt')!r}, "
                    f"got {incoming_prompt!r}"
                )
            self._tolerated = True
        for field, expected, actual in (
            ("task", event.get("task"), got[0]),
            ("agent_id", event.get("agent_id"), got[1]),
            ("system", event.get("system"), got[2]),
            ("temperature", event.get("temperature"), got[3]),
        ):
            if expected != actual:
                mismatch = {
                    "index": index,
                    "kind": "mismatch",
                    "field": field,
                    "expected": expected,
                    "got": actual,
                }
                if self._first_mismatch is None:
                    self._first_mismatch = mismatch
                raise ReplayMismatch(
                    f"replay mismatch at index {index}: field {field!r} expected {expected!r}, got {actual!r}"
                )
        self._cursor += 1
        self._hits += 1
        return event["response"]

    def _after_stop(self, prompt, *args, **kwargs):
        """Post-latch live fall-through — except a PRE-trigger event arriving
        after the window closed: a silent live sample would answer an event
        the protocol promised to replay (kind "late")."""
        latch_stamp = _parse_clock_stamp(self._latch_prompt)
        stamp = _parse_clock_stamp(prompt)
        if latch_stamp is not None and stamp is not None and stamp < latch_stamp:
            mismatch = {
                "index": self._cursor,
                "kind": "late",
                "expected": ">= the trigger sim time",
                "got": prompt,
            }
            if self._first_mismatch is None:
                self._first_mismatch = mismatch
            raise ReplayMismatch(
                f"late pre-trigger call after the replay window closed (latched at "
                f"day {latch_stamp[0]} {latch_stamp[1] // 60:02d}:{latch_stamp[1] % 60:02d}): {prompt!r}"
            )
        return self._base(prompt, *args, **kwargs)

    def coverage(self) -> dict[str, int]:
        return {"hits": self._hits, "total": len(self._events)}

    @property
    def first_mismatch(self) -> dict[str, Any] | None:
        return self._first_mismatch


# the untouched module function, captured before any wrapper is installed
_REAL_CALL_LLM = providers.call_llm


class _OfflineFallback:
    """Default base llm: never the network.

    At call time the fall-through target is whatever ``providers.call_llm``
    currently is — so a wrapper layered on top of this replay (canary, another
    replay) receives the stop-passed call. If nothing else is layered above
    the original provider function captured at install, the call is answered
    with the offline ``"LIVE"`` sentinel: a deterministic replay window must
    not reach a live provider by accident. Callers that genuinely want live
    sampling pass their base explicitly (``base_llm=providers.call_llm`` or a
    stub in tests).
    """

    def __init__(self, original):
        self._target = original

    def __call__(self, prompt, *args, **kwargs):
        # fall through to what was layered here at install time (a canary, a
        # previous replay); if that is the untouched provider itself, answer
        # with the offline sentinel instead of reaching the network
        if self._target is _REAL_CALL_LLM:
            return "LIVE"
        return self._target(prompt, *args, **kwargs)


def install_sequence_replay(
    recorded_events: list[dict[str, Any]],
    base_llm: Callable[..., Any] | None = None,
    stop_predicate: StopPredicate | None = None,
) -> tuple[SequenceReplay, Callable[[], None]]:
    """Rebind ``providers.call_llm`` to an ORDERED replay; return ``(replay, restore)``.

    ``recorded_events`` is an ordered list of
    ``{"task", "agent_id", "system", "temperature", "prompt", "response"}``
    dicts — every recorded call is an event, repeats included, so repeated
    identical prompts replay their own responses in order (the dict replay's
    blind spot). Each incoming request is validated against the event at the
    current position; a mismatch, a reordered call, or an unexpected
    extra/duplicated call raises :class:`ReplayMismatch` naming the exact
    index (and field). ``replay.coverage()`` reports hits over total recorded
    events; ``replay.first_mismatch`` is None or the first failure's record.
    Past the stop predicate (latched) calls fall through to the live base llm
    and consume no event. ``restore()`` unwinds exactly this layer.
    """
    original = providers.call_llm
    if base_llm is None:
        base_llm = _OfflineFallback(original)
    replay = SequenceReplay(recorded_events, base_llm, stop_predicate)

    def restore():
        providers.call_llm = original

    providers.call_llm = replay
    return replay, restore


# (task, agent_id, system, temperature, prompt) -> recorded response
CacheKey = tuple
RecordedResponses = dict[CacheKey, Any]
StopPredicate = Callable[..., bool]


class ReplayCacheMiss(Exception):
    """A pre-stop request has no recorded response.

    Raised, never silently swallowed: falling through to live sampling would
    silently diverge the pre-trigger history that the fork protocol needs.
    """


def install_llm_replay(
    recorded_responses: RecordedResponses,
    base_llm: Callable[..., Any] | None = None,
    stop_predicate: StopPredicate | None = None,
) -> tuple[Callable[..., Any], Callable[[], None]]:
    """Rebind ``providers.call_llm`` to the replay memoizer; return ``(replay, restore)``.

    - HIT (same task / agent_id / system / temperature / prompt, stop
      predicate not yet passed): the recorded response, base llm untouched.
    - MISS before the stop predicate fires: :class:`ReplayCacheMiss` — loud.
    - Stop predicate passed (e.g. past the intervention trigger tick): the
      call goes to the live base llm; fresh sampling resumes.
    - Without a stop predicate every call replays (the documented baseline
      behaviour); predicates are the mechanism that ends the window.

    ``restore()`` puts the original ``providers.call_llm`` back — no global
    state leaks (same guarantee as ``install_canary``).
    """
    original = providers.call_llm
    if base_llm is None:
        base_llm = original

    def replay(prompt, *args, **kwargs):
        if stop_predicate is not None and stop_predicate(prompt, *args, **kwargs):
            return base_llm(prompt, *args, **kwargs)
        key = (
            kwargs.get("task"),
            kwargs.get("agent_id"),
            kwargs.get("system"),
            kwargs.get("temperature"),
            prompt,
        )
        if key not in recorded_responses:
            raise ReplayCacheMiss(
                f"no recorded response for replay key (task={key[0]!r}, agent_id={key[1]!r}, "
                f"system={key[2]!r}, temperature={key[3]!r}, prompt-hash={hash(key[4])})"
            )
        return recorded_responses[key]

    def restore():
        providers.call_llm = original

    providers.call_llm = replay
    return replay, restore
