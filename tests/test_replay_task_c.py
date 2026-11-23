"""Monday Task C: changepoint-aligned replay (cp_joint, cp_per_context)."""

from __future__ import annotations

import pytest

from himher.baselines.hh_agent import best_split, select_changepoint
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.policy import Transition
from himher.experiment import instrumentation as inst
from himher.experiment import rebuttal_methods as rm
from himher.experiment.replay_audit import audit_episode

PREV, ACC = BernoulliHandModel(0.1, 0.1), BernoulliHandModel(0.9, 0.9)
LABELS = [s.label for s in inst.condition_specs()]
W_S = 100


def _window(n_prev: int, n=20):
    """Deterministic draws: theta 0.1 -> the counterpart plays L, 0.9 -> R
    (each model's most likely action); contexts alternate."""
    times = list(range(W_S, W_S + n))
    trs = [Transition(t % 2, Hand.L, Hand.L if i < n_prev else Hand.R, 0.0) for i, t in enumerate(times)]
    return times, trs


@pytest.mark.parametrize("per_context", [False, True])
def test_known_split(per_context):
    times, trs = _window(8)
    sel, changed, s_hat, gain = select_changepoint(times, trs, PREV, ACC, 0.1, per_context)
    assert changed == (0, 1)
    if not per_context:
        assert s_hat["joint"] == W_S + 8
    else:  # first post-change round in each context: 108 (even) and 109 (odd)
        assert s_hat == {"ctx0": W_S + 8, "ctx1": W_S + 9}
    assert [t for t, _ in sel] == list(range(W_S + 8, W_S + 20))
    assert gain > 0


@pytest.mark.parametrize("per_context", [False, True])
def test_no_changed_context_replays_all(per_context):
    times, trs = _window(8)
    sel, changed, _, gain = select_changepoint(times, trs, PREV, PREV, 0.1, per_context)
    assert changed == () and [t for t, _ in sel] == times and [tr for _, tr in sel] == trs and gain == 0


@pytest.mark.parametrize("per_context", [False, True])
def test_window_entirely_after_change(per_context):
    times, trs = _window(0)
    sel, _, s_hat, _ = select_changepoint(times, trs, PREV, ACC, 0.1, per_context)
    if not per_context:
        assert s_hat["joint"] == W_S
    assert [t for t, _ in sel] == times


@pytest.mark.parametrize("per_context", [False, True])
def test_window_entirely_before_change(per_context):
    times, trs = _window(20)
    sel, _, s_hat, _ = select_changepoint(times, trs, PREV, ACC, 0.1, per_context)
    assert all(v == W_S + 20 for v in s_hat.values())  # w_e + 1
    assert sel == []


def test_only_changed_context_is_split():
    prev, acc = BernoulliHandModel(0.1, 0.1), BernoulliHandModel(0.1, 0.9)  # context 0 unchanged
    times, trs = _window(8)
    sel, changed, s_hat, _ = select_changepoint(times, trs, prev, acc, 0.1, False)
    assert changed == (1,)
    kept = [t for t, _ in sel]
    assert all(t in kept for t in times if t % 2 == 0)  # every context-0 interaction
    assert [t for t in kept if t % 2 == 1] == [t for t in times if t % 2 == 1 and t >= s_hat["joint"]]


def test_ties_go_to_the_largest_split():
    # All contributions zero -> every split ties -> s_hat = w_end + 1.
    assert best_split([5, 6, 7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 7) == (8, 0.0)
    assert best_split([], [], [], 9) == (10, 0.0)


@pytest.mark.parametrize("scope", ["cp_joint", "cp_per_context"])
@pytest.mark.parametrize("label", LABELS)
def test_episode_replay_inside_window_and_logged(scope, label):
    spec = inst.condition_by_label(label)
    for seed in range(3):
        audits, _ = audit_episode(f"HH-R@{scope}", label, seed)
        res = inst.run_instrumented(rm.METHODS[f"HH-R@{scope}"].factory(seed, spec), spec, seed, "HH-R", scope, "events")
        acc = [e for e in res.events if e["event"] == "accept"]
        assert len(acc) == len(audits)
        for a, e in zip(audits, acc):
            assert set(a.replay_rounds) <= set(a.him_rounds)
            assert e["n_replayed"] == len(a.replay_rounds) == e["n_replayed_ctx0"] + e["n_replayed_ctx1"]
            assert "changed_contexts" in e and "ll_gain" in e and e["ll_gain"] >= 0
            if e["changed_contexts"] == "none":
                assert a.replay_rounds == a.him_rounds  # identical to full
