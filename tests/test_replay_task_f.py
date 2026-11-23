"""Monday Task F: context_filtered and op_gated replay scopes."""

from __future__ import annotations

import numpy as np
import pytest

from himher.baselines.hh_agent import select_context_filtered, unchanged_contexts
from himher.core.regime import Regime
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.policy import Transition
from himher.experiment import instrumentation as inst
from himher.experiment import rebuttal_methods as rm
from himher.experiment.replay_audit import audit_episode

LABELS = [s.label for s in inst.condition_specs()]


def _buffer(t_end=30, n=20):
    times = list(range(t_end - n + 1, t_end + 1))
    trs = [Transition(t % 2, Hand.L, Hand.R, 0.0) for t in times]
    return times, trs


def test_unchanged_contexts_uses_delta():
    prev = BernoulliHandModel(0.1, 0.1)
    assert unchanged_contexts(prev, BernoulliHandModel(0.9, 0.9), 0.1) == ()
    assert unchanged_contexts(prev, BernoulliHandModel(0.15, 0.6), 0.1) == (0,)
    assert unchanged_contexts(prev, BernoulliHandModel(0.12, 0.08), 0.1) == (0, 1)


def test_both_contexts_change_equals_post_detection():
    times, trs = _buffer()
    sel = select_context_filtered(times, trs, first_detection=25, unchanged=())
    assert [t for t, _ in sel] == [t for t in times if t >= 25]


def test_no_context_changes_equals_full():
    times, trs = _buffer()
    sel = select_context_filtered(times, trs, first_detection=25, unchanged=(0, 1))
    assert [t for t, _ in sel] == times and [tr for _, tr in sel] == trs


def test_one_context_unchanged_keeps_its_pre_detection_transitions():
    times, trs = _buffer()
    sel = [t for t, _ in select_context_filtered(times, trs, first_detection=25, unchanged=(0,))]
    assert sel == [t for t in times if t >= 25 or t % 2 == 0]


def _q(agent):
    return dict(agent.policy._q)


@pytest.mark.parametrize("label", LABELS)
def test_op_gated_matches_full_except_on_reuse(label):
    for seed in range(4):
        full, _ = audit_episode("HH-R", label, seed)
        gated, _ = audit_episode("HH-R@op_gated", label, seed)
        assert [a.t for a in full] == [a.t for a in gated]  # cascade does not see replay
        for f, g in zip(full, gated):
            assert f.regime == g.regime
            if g.regime == "reuse":
                assert g.replay_rounds == ()
            else:
                assert g.replay_rounds == f.replay_rounds


@pytest.mark.parametrize("label", LABELS)
def test_context_filtered_replay_rounds_follow_the_rule(label):
    spec = inst.condition_by_label(label)
    for seed in range(4):
        agent = rm.METHODS["HH-R@context_filtered"].factory(seed, spec)
        audits, _ = audit_episode("HH-R@context_filtered", label, seed)
        res = inst.run_instrumented(agent, spec, seed, "HH-R", "context_filtered", "events")
        acc = [e for e in res.events if e["event"] == "accept"]
        assert len(acc) == len(audits)
        for a, e in zip(audits, acc):
            assert set(a.replay_rounds) <= set(a.him_rounds)  # never outside the HIM window
            assert e["n_replayed"] == len(a.replay_rounds)
            assert e["n_replayed_ctx0"] + e["n_replayed_ctx1"] == e["n_replayed"]
            unchanged = () if e["unchanged_contexts"] == "none" else tuple(map(int, e["unchanged_contexts"].split(";")))
            prev = BernoulliHandModel(e["cur_theta0"], e["cur_theta1"])
            accd = BernoulliHandModel(e["cand_theta0"], e["cand_theta1"])
            assert unchanged == unchanged_contexts(prev, accd, rm.DELTA)


def test_existing_scopes_log_no_task_f_columns():
    spec = inst.condition_by_label("within_library_switch")
    for key in ("HH-R", "HH-R@post_detection"):
        res = inst.run_instrumented(rm.METHODS[key].factory(0, spec), spec, 0, "HH-R", "x", "events")
        assert all("n_replayed_ctx0" not in e for e in res.events)
