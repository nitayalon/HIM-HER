"""Computational-cost counters for the rebuttal (Task C1, Sun 11/22).

Read-only instrumentation: every counted function is wrapped by a thin
pass-through that calls the original with the same arguments and returns
its result unchanged. Counters only increment integers and read
``time.perf_counter``; they never touch an RNG and never change call
order, so returns and model trajectories are bitwise identical with
counting on or off (tests/test_cost_counters.py).

Unit (brief section 3.1): one *model evaluation* = computing
p(counterpart action | context, model) for one model on one observation.
Counted at the source:
  * ``BernoulliHandModel.log_prob`` and ``.predict_proba`` (outermost
    call only -- log_prob calls predict_proba internally, which is not
    counted twice);
  * BPR+'s ``reward_prob`` and ``best_response`` (they read theta
    directly instead of calling the model's methods).
CB / WB / DB evaluate no model at all (conjugate count updates); per the
brief's mapping they are charged 1 analytic evaluation per update
(``ev_monitor``) and 1 per action (``ev_decision``).

Categories are set by wrapping the functions that define them (the
innermost enclosing category gets the evaluation):
  monitor   HHCascade's per-step ``surprise_statistic``; every baseline's
            ``update`` (belief / posterior update)
  revision  ``select_regime`` (reuse retrieval, merge grid, construct) and
            ``HIMVerifier.evaluate`` (also counted separately as
            ``ev_revision_him``); BPR+'s ``_finish_learning``
  voc       ``AlwaysProceedVOC.should_revise`` (evaluates nothing)
  decision  ``TabularQPolicy.act`` (H+H) and every baseline's ``act``
  relabel   ``ReplayBuffer.relabel_and_replay`` (HER)
  policy    ``TabularQPolicy.update`` (one call = one policy update; a
            call made inside relabel is also a replayed sample)
Wall-clock is exclusive per category (time in a nested category is not
also charged to its parent); time inside agent calls outside any
category is ``wall_other_s``.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from himher.baselines import bpr_plus as _bpr
from himher.baselines.bocd_bayes import BOCDBayesAgent
from himher.baselines.bpr_plus import BPRPlusAgent
from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.discounted_bayes import DiscountedBayesAgent
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.baselines.windowed_bayes import WindowedBayesAgent
from himher.core import cascade as _cascade
from himher.core.her import ReplayBuffer
from himher.core.him import HIMVerifier
from himher.core.voc import AlwaysProceedVOC
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.policy import TabularQPolicy

EV_CATS = ("monitor", "revision", "voc", "decision", "relabel")
WALL_CATS = ("monitor", "revision", "voc", "decision", "relabel", "policy", "other")
_EV_OF = {"monitor": "monitor", "revision": "revision", "revision_him": "revision", "voc": "voc",
          "decision": "decision", "relabel": "relabel"}
_WALL_OF = {"revision_him": "revision"}


class CostCounter:
    def __init__(self) -> None:
        self.ev = dict.fromkeys(EV_CATS, 0)
        self.ev_revision_him = 0
        self.ev_unattributed = 0
        self.cache_hits = 0
        self.pol_updates = 0
        self.pol_samples_replayed = 0
        self.n_revision_triggers = 0
        self.n_candidates = 0
        self.wall = dict.fromkeys(WALL_CATS, 0.0)
        self._stack: list[str] = []
        self._t_last = 0.0

    # -- bookkeeping ------------------------------------------------------
    @property
    def ev_total(self) -> int:
        return sum(self.ev.values())

    def _credit(self, now: float) -> None:
        if self._stack:
            top = self._stack[-1]
            if top != "paused":
                self.wall[_WALL_OF.get(top, top)] += now - self._t_last
        self._t_last = now

    def push(self, cat: str) -> None:
        self._credit(time.perf_counter())
        self._stack.append(cat)

    def pop(self) -> None:
        self._credit(time.perf_counter())
        self._stack.pop()

    @property
    def top(self) -> str | None:
        return self._stack[-1] if self._stack else None

    def add_eval(self, n: int = 1) -> None:
        for cat in reversed(self._stack):
            if cat == "paused":
                return
            ev_cat = _EV_OF.get(cat)
            if ev_cat is not None:
                self.ev[ev_cat] += n
                if cat == "revision_him":
                    self.ev_revision_him += n
                return
        self.ev_unattributed += n

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {f"ev_{k}": v for k, v in self.ev.items()}
        d.update(ev_total=self.ev_total, ev_revision_him=self.ev_revision_him,
                 ev_revision_generate=self.ev["revision"] - self.ev_revision_him,
                 ev_unattributed=self.ev_unattributed, cache_hits=self.cache_hits,
                 pol_updates=self.pol_updates, pol_samples_replayed=self.pol_samples_replayed,
                 n_revision_triggers=self.n_revision_triggers, n_candidates=self.n_candidates)
        d.update({f"wall_{k}_s": v for k, v in self.wall.items()})
        return d


ACTIVE: CostCounter | None = None
_depth = 0  # >0 while inside a counted model evaluation (no double counting)
_ORIGINALS: list[tuple[Any, str, Any]] = []


def _wrap_eval(orig: Callable) -> Callable:
    def counted(*args: Any, **kwargs: Any) -> Any:
        global _depth
        c = ACTIVE
        if c is None or _depth:
            return orig(*args, **kwargs)
        _depth += 1
        try:
            return orig(*args, **kwargs)
        finally:
            _depth -= 1
            c.add_eval(1)
    counted.__wrapped__ = orig  # type: ignore[attr-defined]
    return counted


def _wrap_cat(orig: Callable, cat: str, on_enter: Callable[[CostCounter, tuple], None] | None = None) -> Callable:
    def categorised(*args: Any, **kwargs: Any) -> Any:
        c = ACTIVE
        if c is None:
            return orig(*args, **kwargs)
        if on_enter is not None:
            on_enter(c, args)
        c.push(cat)
        try:
            return orig(*args, **kwargs)
        finally:
            c.pop()
    categorised.__wrapped__ = orig  # type: ignore[attr-defined]
    return categorised


def _on_trigger(c: CostCounter, _args: tuple) -> None:
    c.n_revision_triggers += 1


def _on_candidate(c: CostCounter, _args: tuple) -> None:
    c.n_candidates += 1


def _on_policy_update(c: CostCounter, _args: tuple) -> None:
    c.pol_updates += 1
    if c.top == "relabel":
        c.pol_samples_replayed += 1


def _bocd_cache_hits(c: CostCounter, args: tuple) -> None:
    agent = args[0]
    # The |H| likelihoods computed this step are reused by every live run
    # length (growth term and changepoint predictive).
    c.cache_hits += agent._run_log_likelihoods.shape[0] * len(agent._hypotheses)


def _analytic(cat: str, orig: Callable) -> Callable:
    """CB/WB/DB: one logical evaluation per update / per action (brief 3.3)."""
    def charged(*args: Any, **kwargs: Any) -> Any:
        c = ACTIVE
        if c is None:
            return orig(*args, **kwargs)
        c.push(cat)
        try:
            c.add_eval(1)
            return orig(*args, **kwargs)
        finally:
            c.pop()
    charged.__wrapped__ = orig  # type: ignore[attr-defined]
    return charged


def _patch(owner: Any, name: str, new: Callable) -> None:
    _ORIGINALS.append((owner, name, owner.__dict__[name]))
    setattr(owner, name, new)


def install() -> None:
    """Wrap every counted function (idempotent). With ACTIVE = None the
    wrappers are pure pass-throughs."""
    if _ORIGINALS:
        return
    # Model evaluations.
    _patch(BernoulliHandModel, "log_prob", _wrap_eval(BernoulliHandModel.log_prob))
    _patch(BernoulliHandModel, "predict_proba", _wrap_eval(BernoulliHandModel.predict_proba))
    _patch(_bpr, "reward_prob", _wrap_eval(_bpr.reward_prob))
    _patch(_bpr, "best_response", _wrap_eval(_bpr.best_response))
    # H+H categories.
    _patch(_cascade, "surprise_statistic", _wrap_cat(_cascade.surprise_statistic, "monitor"))
    _patch(_cascade, "select_regime", _wrap_cat(_cascade.select_regime, "revision", _on_trigger))
    _patch(HIMVerifier, "evaluate", _wrap_cat(HIMVerifier.evaluate, "revision_him", _on_candidate))
    _patch(AlwaysProceedVOC, "should_revise", _wrap_cat(AlwaysProceedVOC.should_revise, "voc"))
    _patch(ReplayBuffer, "relabel_and_replay", _wrap_cat(ReplayBuffer.relabel_and_replay, "relabel"))
    _patch(TabularQPolicy, "update", _wrap_cat(TabularQPolicy.update, "policy", _on_policy_update))
    _patch(TabularQPolicy, "act", _wrap_cat(TabularQPolicy.act, "decision"))
    # Baselines whose updates/actions call model methods (counted at source).
    for cls in (FixedLibraryBayesAgent, BPRPlusAgent):
        _patch(cls, "update", _wrap_cat(cls.update, "monitor"))
        _patch(cls, "act", _wrap_cat(cls.act, "decision"))
    _patch(BOCDBayesAgent, "update", _wrap_cat(BOCDBayesAgent.update, "monitor", _bocd_cache_hits))
    _patch(BOCDBayesAgent, "act", _wrap_cat(BOCDBayesAgent.act, "decision"))
    _patch(BPRPlusAgent, "_finish_learning", _wrap_cat(BPRPlusAgent._finish_learning, "revision"))
    # Conjugate-count baselines: analytic 1 + 1 per step.
    for cls in (ContinuousBayesAgent, WindowedBayesAgent, DiscountedBayesAgent):
        _patch(cls, "update", _analytic("monitor", cls.update))
        _patch(cls, "act", _analytic("decision", cls.act))


def uninstall() -> None:
    while _ORIGINALS:
        owner, name, orig = _ORIGINALS.pop()
        setattr(owner, name, orig)


class counting:
    """Context manager: install wrappers and make ``counter`` active."""

    def __init__(self, counter: CostCounter) -> None:
        self.counter = counter

    def __enter__(self) -> CostCounter:
        global ACTIVE
        install()
        ACTIVE = self.counter
        return self.counter

    def __exit__(self, *exc: Any) -> None:
        global ACTIVE
        ACTIVE = None


class paused:
    """Suppress counting (e.g. for logging reads) inside an active run."""

    def __enter__(self) -> None:
        if ACTIVE is not None:
            ACTIVE.push("paused")

    def __exit__(self, *exc: Any) -> None:
        if ACTIVE is not None:
            ACTIVE.pop()
