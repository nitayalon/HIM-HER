import pytest

from himher.baselines.hh_agent import decay_reset_epsilon
from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import METHODS, make_hh_agent
from himher.experiment.runner import run_episode

LABELS = ["stationary", "within_library_switch", "outside_library_switch_m1", "recurring"]


@pytest.mark.parametrize("label", LABELS)
def test_constant_schedule_matches_pre_change_default(label):
    spec = inst.condition_by_label(label)
    for seed in range(5):
        a = make_hh_agent(seed, replay=True)
        b = make_hh_agent(seed, replay=True, eps_schedule="constant")
        ra = [l.reward for l in run_episode(a, spec.condition, seed)]
        rb = [l.reward for l in run_episode(b, spec.condition, seed)]
        assert ra == rb
        assert dict(a.policy._q) == dict(b.policy._q)
        assert a.policy.epsilon == b.policy.epsilon == 0.15


def test_decay_formula_endpoints():
    assert decay_reset_epsilon(37, 37, 0.15, 0.02, 50) == 0.15
    assert decay_reset_epsilon(37 + 50, 37, 0.15, 0.02, 50) == pytest.approx(0.075)
    assert decay_reset_epsilon(10_000, 0, 0.15, 0.02, 50) == 0.02
    # floor reached at t - t_reset = 50 * log2(0.15 / 0.02) ~ 145.4
    assert decay_reset_epsilon(145, 0, 0.15, 0.02, 50) > 0.02
    assert decay_reset_epsilon(146, 0, 0.15, 0.02, 50) == 0.02


@pytest.mark.parametrize("label", LABELS)
def test_decay_reset_trace_matches_formula_and_resets_on_every_detection(label):
    spec = inst.condition_by_label(label)
    for seed in range(5):
        agent = METHODS["HH-R@decay_reset"].factory(seed, spec)
        res = inst.run_instrumented(agent, spec, seed, "HH-R", "decay_reset", log_level="full")
        eps = res.per_step["epsilon"]
        detected = [s.detected for s in agent.step_logs]
        assert agent.eps_reset_times == [0] + [t for t, d in enumerate(detected) if d]
        t_reset = 0
        for t in range(len(eps)):
            assert eps[t] == decay_reset_epsilon(t, t_reset, 0.15, 0.02, 50.0)
            if detected[t]:
                t_reset = t
        assert eps[0] == 0.15
        resets = [e for e in res.events if e["event"] == "eps_reset"]
        assert [e["t"] for e in resets] == agent.eps_reset_times


def test_decay_reset_does_not_change_the_cascade():
    """Ego exploration cannot affect detection/HIM (they only see the
    environment's draws), so acceptance rounds are paired exactly."""
    spec = inst.condition_by_label("within_library_switch")
    for seed in range(5):
        a = make_hh_agent(seed, replay=True)
        b = METHODS["HH-R@decay_reset"].factory(seed, spec)
        run_episode(a, spec.condition, seed)
        run_episode(b, spec.condition, seed)
        assert [s.incumbent_after for s in a.step_logs] == [s.incumbent_after for s in b.step_logs]


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError):
        make_hh_agent(0, replay=True, eps_schedule="cosine")
