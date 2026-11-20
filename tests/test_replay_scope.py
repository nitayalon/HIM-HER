import pytest

from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import make_hh_agent
from himher.experiment.runner import run_episode

LABELS = ["stationary", "within_library_switch", "outside_library_switch_m2", "recurring"]


def _rewards(agent, spec, seed):
    return [l.reward for l in run_episode(agent, spec.condition, seed)]


def _q_table(agent):
    return dict(agent.policy._q)


@pytest.mark.parametrize("label", LABELS)
def test_full_scope_matches_pre_change_default(label):
    """replay_scope='full' is the default and must equal an agent built
    without the flag (i.e. the paper's HH-R)."""
    spec = inst.condition_by_label(label)
    for seed in range(5):
        a = make_hh_agent(seed, replay=True)
        b = make_hh_agent(seed, replay=True, replay_scope="full")
        assert _rewards(a, spec, seed) == _rewards(b, spec, seed)
        assert _q_table(a) == _q_table(b)


@pytest.mark.parametrize("label", LABELS)
def test_him_window_replays_exactly_the_him_window(label):
    spec = inst.condition_by_label(label)
    for seed in range(5):
        agent = make_hh_agent(seed, replay=True, replay_scope="him_window")
        res = inst.run_instrumented(agent, spec, seed, "HH-R", "him_window", log_level="events")
        accepts = [e for e in res.events if e["event"] == "accept"]
        assert len(accepts) == len(agent.replay_log) == res.summary["n_revisions"]
        for e in accepts:
            assert e["n_replayed"] == e["window_end"] - e["window_start"] + 1
            assert (e["replay_start"], e["replay_end"]) == (e["window_start"], e["window_end"])


@pytest.mark.parametrize("label", LABELS)
def test_him_window_coincides_with_full_in_this_codebase(label):
    """The HER buffer and HIM's history are both the last k observations,
    so the two scopes select identical transitions (documented finding)."""
    spec = inst.condition_by_label(label)
    for seed in range(5):
        a = make_hh_agent(seed, replay=True, replay_scope="full")
        b = make_hh_agent(seed, replay=True, replay_scope="him_window")
        assert _rewards(a, spec, seed) == _rewards(b, spec, seed)
        assert a.replay_log == b.replay_log


def test_post_detection_replays_from_detection_run_start():
    spec = inst.condition_by_label("outside_library_switch_m2")
    for seed in range(5):
        agent = make_hh_agent(seed, replay=True, replay_scope="post_detection")
        run_episode(agent, spec.condition, seed)
        logs = agent.step_logs
        for t, n, first, last in agent.replay_log:
            assert last == t and n == t - first + 1
            # every round from `first` to t was a detection, the round before was not
            # (or was itself an acceptance, which starts a new run), or the run
            # was truncated by the k-sized buffer
            assert all(logs[u].detected for u in range(first, t + 1))
            truncated = n == 20
            assert truncated or first == 0 or not logs[first - 1].detected or logs[first - 1].revised


def test_post_detection_never_replays_more_than_full():
    spec = inst.condition_by_label("within_library_switch")
    for seed in range(5):
        full = make_hh_agent(seed, replay=True)
        post = make_hh_agent(seed, replay=True, replay_scope="post_detection")
        run_episode(full, spec.condition, seed)
        run_episode(post, spec.condition, seed)
        # cascade is independent of ego actions -> same acceptance rounds
        assert [r[0] for r in full.replay_log] == [r[0] for r in post.replay_log]
        assert all(p[1] <= f[1] for f, p in zip(full.replay_log, post.replay_log))


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        make_hh_agent(0, replay=True, replay_scope="everything")
