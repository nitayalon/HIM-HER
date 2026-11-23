"""Tests for analysis/replay_diagnostic.py (brief: replay-contamination)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "replay_diagnostic", Path(__file__).resolve().parents[1] / "analysis" / "replay_diagnostic.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def _ll(theta, c, a):
    p = theta[c] if a == 1 else 1 - theta[c]
    return math.log(p)


def test_synthetic_hand_computed():
    # Switch at t_sw = 4; buffer replays rounds 2..6 (5 transitions, 2 pre-switch).
    contexts = [0, 1, 0, 1, 1, 0, 1]
    actions = [0, 0, 1, 0, 1, 1, 1]
    true_pre, cand, cur = (0.1, 0.1), (0.9, 0.9), (0.2, 0.3)
    m = rd.acceptance_metrics([2, 3, 4, 5, 6], 4, contexts, actions, cand, cur, true_pre)
    assert m["n_R"] == 5 and m["n_R_pre"] == 2 and m["pre_share"] == pytest.approx(0.4)
    pre = [(0, 1), (1, 0)]  # (context, action) of rounds 2, 3
    for key, th in (("ll_pre_accepted", cand), ("ll_pre_prev", cur), ("ll_pre_true", true_pre)):
        assert m[key] == pytest.approx(np.mean([_ll(th, c, a) for c, a in pre]))
    assert m["gap_pre"] == pytest.approx(m["ll_pre_accepted"] - m["ll_pre_true"])
    post = [(1, 1), (0, 1), (1, 1)]
    assert m["ll_post_accepted"] == pytest.approx(np.mean([_ll(cand, c, a) for c, a in post]))
    # per-context restriction: context 0 -> rounds 2 (pre) and 5 (post)
    m0 = rd.acceptance_metrics([2, 3, 4, 5, 6], 4, contexts, actions, cand, cur, true_pre, context=0)
    assert m0["n_R"] == 2 and m0["n_R_pre"] == 1
    assert m0["ll_pre_true"] == pytest.approx(_ll(true_pre, 0, 1))


def test_him_window_after_switch_has_zero_pre_share():
    T = 200
    detected = np.zeros(T, bool)
    t_sw, t_acc = 150, 175  # window = rounds 156..175, entirely after the switch
    R = rd.replay_rounds("him_window", t_acc, detected, {t_acc})
    assert R == list(range(156, 176))
    m = rd.acceptance_metrics(R, t_sw, [0] * T, [0] * T, (0.5, 0.5), (0.5, 0.5), (0.1, 0.1))
    assert m["pre_share"] == 0 and math.isnan(m["ll_pre_accepted"])


def test_post_detection_run_reconstruction():
    detected = np.zeros(60, bool)
    detected[40:51] = True  # detection run 40..50, acceptance at 45 restarts the run
    assert rd.replay_rounds("post_detection", 45, detected, {45, 50}) == list(range(40, 46))
    assert rd.replay_rounds("post_detection", 50, detected, {45, 50}) == list(range(46, 51))
    assert rd.replay_rounds("full", 50, detected, {50}) == list(range(31, 51))


def test_swapping_models_swaps_likelihoods():
    contexts, actions = [0, 1, 0, 1, 0], [1, 0, 0, 1, 1]
    a = rd.acceptance_metrics(range(5), 3, contexts, actions, (0.7, 0.2), (0.3, 0.6), (0.1, 0.1))
    b = rd.acceptance_metrics(range(5), 3, contexts, actions, (0.3, 0.6), (0.7, 0.2), (0.1, 0.1))
    assert a["ll_pre_accepted"] == b["ll_pre_prev"] and a["ll_pre_prev"] == b["ll_pre_accepted"]
    assert a["ll_post_accepted"] == b["ll_post_prev"]
