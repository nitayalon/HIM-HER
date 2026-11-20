import math

import numpy as np
import pytest

from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import METHODS
from himher.experiment.runner import run_episode

SEEDS = range(5)
RETURN_KEYS = ("return_pre", "return_post", "return_total", "auc", "regret_post", "recovery_time",
               "kld_final", "n_detections", "n_accept_reuse", "n_accept_merge", "n_accept_construct",
               "n_reject", "n_policy_updates")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.mark.parametrize("key", ["HH-R", "HH-A", "CB", "FB"])
@pytest.mark.parametrize("label", ["within_library_switch", "outside_library_switch_m2"])
def test_log_levels_give_bitwise_identical_results(key, label):
    spec = inst.condition_by_label(label)
    for seed in SEEDS:
        rows = []
        for level in inst.LOG_LEVELS:
            agent = METHODS[key].factory(seed, spec)
            rows.append(inst.run_instrumented(agent, spec, seed, key, log_level=level).summary)
        for k in RETURN_KEYS:
            assert _same(rows[0][k], rows[1][k]) and _same(rows[0][k], rows[2][k]), k


@pytest.mark.parametrize("key", ["HH-R", "HH-A", "CB", "OB"])
def test_instrumented_loop_matches_run_episode(key):
    spec = inst.condition_by_label("recurring")
    for seed in range(3):
        logs = run_episode(METHODS[key].factory(seed, spec), spec.condition, seed)
        res = inst.run_instrumented(METHODS[key].factory(seed, spec), spec, seed, key, log_level="full")
        np.testing.assert_array_equal(res.per_step["ego_reward"], [l.reward for l in logs])
        np.testing.assert_array_equal(res.per_step["ego_action"], [int(l.ego_action) for l in logs])
        np.testing.assert_array_equal(res.per_step["context"], [l.context for l in logs])


def test_event_rows_are_consistent():
    spec = inst.condition_by_label("outside_library_switch_m2")
    agent = METHODS["HH-R"].factory(0, spec)
    res = inst.run_instrumented(agent, spec, 0, "HH-R", log_level="events")
    accepts = [e for e in res.events if e["event"] == "accept"]
    rejects = [e for e in res.events if e["event"] == "reject"]
    cands = [e for e in res.events if e["event"] == "candidate"]
    assert len(accepts) == res.summary["n_revisions"]
    assert len(rejects) == res.summary["n_reject"]
    assert len(cands) == len(accepts) + len(rejects)
    assert sum(e["event"] == "detection" for e in res.events) == res.summary["n_detections"]
    for e in accepts:
        assert e["delta_loglik"] > 1.0  # tau, strict
        assert e["min_distinctness"] >= 0.1
        assert e["window_end"] == e["t"] and e["window_end"] - e["window_start"] + 1 <= 20


def test_kl_matches_paper_definition():
    from himher.domains.handoff.counterpart import BernoulliHandModel
    true = BernoulliHandModel(0.1, 0.1)
    # stored kld_raw.csv, stationary/hh_replay/seed 0/t=1: model (0.1, 0.9)
    assert inst.context_avg_kl(true, (0.1, 0.9)) == pytest.approx(0.8788898309344879, abs=1e-12)
