"""HER replay never reaches outside HIM's verification window.

Exact boundaries (window k = 20), for an acceptance at round t:
  HIM window W = rounds max(0, t - k + 1) .. t   (inclusive; includes round t)
  HH-R replay  R = W exactly, same order           (buffer and history are
                                                   both the last k rounds)
  HH-A replay  R = {}                              (buffer discarded)
  HH-R@him_window / @post_detection: R subset of W
Rounds are identified by object identity of the replayed transitions and by
the window actually passed to HIMVerifier.evaluate, not by assumption.
"""

from __future__ import annotations

import pytest

from himher.experiment import instrumentation as inst
from himher.experiment.cost_runner import make_agent, run_cost
from himher.experiment.replay_audit import audit_episode

K = 20
LABELS = [s.label for s in inst.condition_specs()]
SEEDS = range(10)


def _expected_window(t: int) -> tuple[int, ...]:
    return tuple(range(max(0, t - K + 1), t + 1))


@pytest.mark.parametrize("label", LABELS)
def test_hhr_replay_equals_him_window(label):
    n = 0
    for seed in SEEDS:
        audits, _ = audit_episode("HH-R", label, seed)
        for a in audits:
            n += 1
            assert a.him_window_matches
            assert a.him_rounds == _expected_window(a.t)
            assert a.replay_rounds == a.him_rounds
            assert min(a.replay_rounds) >= min(a.him_rounds)  # nothing predates W
            assert max(a.replay_rounds) == a.t                 # nothing after the acceptance
    assert n > 0


@pytest.mark.parametrize("label", LABELS)
def test_hha_replays_nothing(label):
    n = 0
    for seed in SEEDS:
        audits, _ = audit_episode("HH-A", label, seed)
        n += len(audits)
        assert all(a.replay_rounds == () for a in audits)
    assert n > 0


@pytest.mark.parametrize("method", ["HH-R@him_window", "HH-R@post_detection"])
@pytest.mark.parametrize("label", LABELS)
def test_scoped_replay_inside_him_window(method, label):
    for seed in SEEDS:
        for a in audit_episode(method, label, seed)[0]:
            assert a.him_window_matches
            assert set(a.replay_rounds) <= set(a.him_rounds)
            assert a.replay_rounds and min(a.replay_rounds) >= min(a.him_rounds)


@pytest.mark.parametrize("method", ["HH-R", "HH-A", "HH-R@post_detection"])
def test_audit_is_read_only(method):
    for label in LABELS:
        spec = inst.condition_by_label(label)
        for seed in range(3):
            _, rewards = audit_episode(method, label, seed)
            plain = run_cost(make_agent(method, seed, spec), spec, seed, count=False)["extras"]["rewards"]
            assert rewards == plain


def test_him_window_can_contain_previous_partner_behaviour():
    """Documents (does not endorse) that passing HIM does not make every
    replayed transition come from the accepted partner behaviour: the
    window straddles the switch, and HH-R replays all of it."""
    found = [a for seed in SEEDS for a in audit_episode("HH-R", "within_library_switch", seed)[0]
             if a.replay_prev_behaviour > 0]
    assert found
    assert all(a.replay_prev_behaviour == a.him_prev_behaviour for a in found)
