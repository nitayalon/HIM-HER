"""Task C2: verification of the cost counters."""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from himher.baselines.bocd_bayes import BOCDBayesAgent
from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.experiment import conditions as cond
from himher.experiment import cost_counters as cc
from himher.experiment import instrumentation as inst
from himher.experiment.cost_runner import COST_METHODS, make_agent, run_cost
from himher.experiment.rebuttal_methods import make_hh_agent
from himher.experiment.runner import run_episode

LABELS = [s.label for s in inst.condition_specs()]


def _spec(T: int) -> inst.ConditionSpec:
    return inst.ConditionSpec("stationary", cond.stationary(episode_length=T))


# 1. Bitwise invariance ------------------------------------------------------
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("key", COST_METHODS)
def test_counting_is_bitwise_invisible(key, label):
    spec = inst.condition_by_label(label)
    for seed in range(5):
        cc.uninstall()  # pristine code path: no wrappers at all
        off = run_cost(make_agent(key, seed, spec), spec, seed, count=False, record_trajectory=True)
        plain = [l.reward for l in run_episode(make_agent(key, seed, spec), spec.condition, seed)]
        on = run_cost(make_agent(key, seed, spec), spec, seed, count=True, record_trajectory=True)
        assert off["extras"]["rewards"] == plain == on["extras"]["rewards"]
        assert off["extras"]["actions"] == on["extras"]["actions"]
        assert off["extras"]["trajectory"] == on["extras"]["trajectory"]
        assert on["row"]["ev_unattributed"] == 0
        for k in ("return_pre", "return_total", "auc", "n_detections"):
            a, b = off["row"][k], on["row"][k]
            assert (a == b) or (np.isnan(a) and np.isnan(b))


def test_pass_through_when_inactive():
    cc.install()
    spec = inst.condition_by_label("within_library_switch")
    a = [l.reward for l in run_episode(make_agent("HH-R", 3, spec), spec.condition, 3)]
    cc.uninstall()
    b = [l.reward for l in run_episode(make_agent("HH-R", 3, spec), spec.condition, 3)]
    assert a == b


# 2. Analytic counts (100-step Stationary) -----------------------------------
def test_cb_counts():
    r = run_cost(ContinuousBayesAgent(alpha=1.0), _spec(100), 0)["row"]
    assert (r["ev_monitor"], r["ev_decision"], r["ev_revision"], r["ev_total"]) == (100, 100, 0, 200)


def test_fb_counts():
    L = len(cond.LIBRARY_0)
    r = run_cost(FixedLibraryBayesAgent(cond.LIBRARY_0), _spec(100), 0)["row"]
    assert r["ev_monitor"] == 100 * L and r["ev_decision"] == 100 * L


def test_bocpd_counts():
    L = len(cond.LIBRARY_0)
    r = run_cost(BOCDBayesAgent(cond.LIBRARY_0, hazard=0.01), _spec(100), 0)["row"]
    assert r["ev_monitor"] == 100 * L and r["ev_decision"] == 100 * L
    # update t (1-indexed) sees t live run lengths, R_max = inf (no pruning)
    assert r["cache_hits"] == L * sum(min(t, 10**9) for t in range(1, 101))


def test_hh_monitor_is_window_rescoring():
    r = run_cost(make_hh_agent(0, replay=True), _spec(100), 0)["row"]
    assert r["ev_monitor"] == sum(min(t, 20) for t in range(1, 101))
    assert r["ev_decision"] == 0


# 3. H+H revision accounting ------------------------------------------------
def test_no_detection_means_no_revision_cost():
    agent = make_hh_agent(0, replay=True)
    agent.cascade.kappa = 1e9  # test-only: never detect
    r = run_cost(agent, inst.condition_by_label("within_library_switch"), 0)["row"]
    assert r["n_detections"] == 0
    assert r["ev_revision"] == r["ev_voc"] == r["ev_relabel"] == 0
    assert r["pol_updates"] == 300 and r["pol_samples_replayed"] == 0


def expected_revision_evals(incumbent, library, window_len, attempts):
    """Exact count for one select_regime call.
      reuse retrieval: |lib| * h
      merge grid:      C(|lib|, 2) * 21 * h        (if |lib| >= 2)
      per HIM test:    (1 + |C|) * h  log-likelihoods
                       + 2 * |C| * |contexts|  predict_proba (distinctness),
      with C = {incumbent} + (lib minus incumbent and candidate)."""
    lib = list(library)
    n = len(lib) * window_len  # reuse retrieval always runs
    # merge grid runs unless reuse was accepted first
    reuse_accepted = attempts and attempts[0].regime.value == "reuse" and attempts[0].result.accepted
    if not reuse_accepted and len(lib) >= 2:
        n += comb(len(lib), 2) * 21 * window_len
    for a in attempts:
        pool = [m for m in lib if m != incumbent and m != a.model]
        c = 1 + len(pool)
        n += (1 + c) * window_len + 2 * c * 2
    return n


@pytest.mark.parametrize("label", ["within_library_switch", "outside_library_switch_m2", "recurring"])
def test_revision_count_matches_formula_for_every_trigger(label):
    spec = inst.condition_by_label(label)
    for seed in range(3):
        agent = make_hh_agent(seed, replay=True)
        counter = cc.CostCounter()
        game_rng = np.random.default_rng(seed)
        from himher.domains.handoff.env import SwitchingHandoffGame
        game = SwitchingHandoffGame(spec.condition.schedule, game_rng)
        checked = 0
        with cc.counting(counter):
            for t in range(spec.condition.episode_length):
                x = game.sample_context()
                a = agent.act(x)
                res = game.step(x, a)
                with cc.paused():
                    lib_before = tuple(agent.cascade.library)
                    inc_before = agent.cascade.incumbent
                    win = min(len(agent.cascade.history) + 1, 20)
                before = counter.ev["revision"]
                agent.update(x, a, res.counterpart_action, res.reward)
                log = agent.step_logs[-1]
                spent = counter.ev["revision"] - before
                if log.detected:
                    assert spent == expected_revision_evals(inc_before, lib_before, win, log.attempts)
                    checked += 1
                else:
                    assert spent == 0
        assert checked > 0


# 4. Consistency with the existing instrumentation ---------------------------
@pytest.mark.parametrize("label", LABELS)
def test_event_counts_match_instrumentation(label):
    spec = inst.condition_by_label(label)
    for key in ("HH-R", "HH-A"):
        for seed in range(3):
            r = run_cost(make_agent(key, seed, spec), spec, seed)["row"]
            ag = make_agent(key, seed, spec)
            res = inst.run_instrumented(ag, spec, seed, key, log_level="events")
            s = res.summary
            assert r["n_revision_triggers"] == r["n_detections"] == s["n_detections"]
            assert r["n_candidates"] == sum(1 for e in res.events if e["event"] == "candidate")
            assert r["pol_updates"] == s["n_policy_updates"]
            for op in ("reuse", "merge", "construct"):
                assert r[f"n_accept_{op}"] == s[f"n_accept_{op}"]
            assert r["n_reject"] == s["n_reject"]


def test_bpr_triggers_match_detections():
    spec = inst.condition_by_label("outside_library_switch_m2")
    r = run_cost(make_agent("BPR+", 1, spec), spec, 1)["row"]
    ag = make_agent("BPR+", 1, spec)
    s = inst.run_instrumented(ag, spec, 1, "BPR+").summary
    assert r["n_revision_triggers"] == s["n_detections"]
    assert r["n_candidates"] == s["n_revisions"]
