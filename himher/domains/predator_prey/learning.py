"""Learned H+H predator for the Predator-Prey game: model-conditioned Q-learning
with HER (paper Sections 4.3 and 5.2, Algorithm 1 line 20).

Q(s, m, a) = w_m . phi(s, a): one weight vector per counterpart model m (the
Section 5.2 table, with a linear function in place of a lookup table, since
there are 2,401 positions). The features are fixed value functions that never
look at m, so a new model's Q really has to be learned -- that is the gap HER
closes.

On acceptance of m' (HIM), following Algorithm 1 line 20:
  * m' starts from the incumbent's weights if it has none of its own (Section
    4.3: fine-tune from the current policy), and keeps its own if it was used
    before (Reuse);
  * with replay (HH-R), the buffered window -- the transitions HIM just checked
    m' against (Remark 1) -- is relabeled with m' and replayed, with step size
    proportional to d(m, m') over the window;
  * without replay (HH-A), the buffer is discarded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from himher.core.cascade import CascadeStepLog, HHCascade
from himher.core.her import ReplayBuffer
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.model import model_divergence
from himher.domains.predator_prey.agents import LIBRARY_0
from himher.domains.predator_prey.construct import construct_weight_model
from himher.domains.predator_prey.counterpart import DEN_A_MODEL, DEN_B_MODEL, WeightedPreyModel
from himher.domains.predator_prey.env import N_PRED_ACTIONS, T, Context
from himher.domains.predator_prey.planner import plan, plan_worst_case
from himher.domains.predator_prey.revise import revise_weight_mixture

FEATURE_NAMES = ("bias", "value_vs_den_A", "value_vs_den_B", "value_vs_worst_case")


def _build_features() -> np.ndarray:
    """PHI[t, p, q, a, k]: the predator's planned values for move a at round t
    against the two library models it knows from the start (Den A, Den B) and
    against a worst-case prey (a model-free pursuit value). Q for any model is
    learned as a weighting of these -- linear value transfer from known
    policies, in the spirit of generalized policy improvement. No held-out
    type (Evade, Cautious-B) is used."""
    basis = [plan(DEN_A_MODEL), plan(DEN_B_MODEL), plan_worst_case()]
    return np.stack([np.ones_like(basis[0]), *basis], axis=-1)


PHI = _build_features()


@dataclass(frozen=True)
class Transition:
    t: int
    context: Context
    action: int
    reward: float
    next_context: Context
    done: bool


class LinearModelQ:
    """Q(s, m, a) = w_m . phi(s, a), learned by semi-gradient Q-learning."""

    def __init__(self, learning_rate: float, gamma: float) -> None:
        self.lr = learning_rate
        self.gamma = gamma
        self.weights: dict[WeightedPreyModel, np.ndarray] = {}
        self.update_counts: dict[WeightedPreyModel, int] = defaultdict(int)

    def ensure(self, model: WeightedPreyModel, init_from: WeightedPreyModel | None = None) -> None:
        if model not in self.weights:
            source = self.weights.get(init_from) if init_from is not None else None
            self.weights[model] = (
                source.copy() if source is not None else np.zeros(len(FEATURE_NAMES))
            )

    def q_row(self, t: int, context: Context, model: WeightedPreyModel) -> np.ndarray:
        p, q = context
        return PHI[t, p, q] @ self.weights[model]

    def update(self, x: Transition, model: WeightedPreyModel, lr: float | None = None) -> None:
        w = self.weights[model]
        target = x.reward
        if not x.done:
            target += self.gamma * float(np.max(self.q_row(x.t + 1, x.next_context, model)))
        p, q = x.context
        features = PHI[x.t, p, q, x.action]
        w += (self.lr if lr is None else lr) * (target - features @ w) * features
        self.update_counts[model] += 1


class LearnedHHPredator:
    def __init__(
        self,
        cascade: HHCascade,
        rng: np.random.Generator,
        replay: bool,
        epsilon: float,
        learning_rate: float,
        gamma: float,
        replay_epochs: int,
        warm_start: str = "nearest",
    ) -> None:
        self._warm_start = warm_start
        self.cascade = cascade
        self.q = LinearModelQ(learning_rate, gamma)
        # The starting library comes with its responses: Q for Den A / Den B is
        # exactly their own basis value function (as a policy-reuse library
        # stores a response policy with each type). Models built later start
        # from the incumbent's weights and must be learned.
        for k, model in ((1, DEN_A_MODEL), (2, DEN_B_MODEL)):
            self.q.weights[model] = np.eye(len(FEATURE_NAMES))[k].copy()
        self.q.ensure(cascade.incumbent)
        self._rng = rng
        self._replay = replay
        self._epsilon = epsilon
        self._replay_epochs = replay_epochs
        self._buffer: ReplayBuffer[Transition] = ReplayBuffer(maxlen=cascade.window)
        self._pending: tuple[WeightedPreyModel, CascadeStepLog] | None = None
        self.step_logs: list[CascadeStepLog] = []
        self.replay_updates = 0

    def act(self, context: Context, t: int) -> int:
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(N_PRED_ACTIONS))
        row = self.q.q_row(t, context, self.cascade.incumbent)
        best = np.flatnonzero(row >= row.max() - 1e-12)
        return int(best[0] if len(best) == 1 else self._rng.choice(best))

    def observe(self, context: Context, prey_action: int) -> None:
        model_at_action_time = self.cascade.incumbent
        log = self.cascade.step(context, prey_action)
        self.step_logs.append(log)
        self._pending = (model_at_action_time, log)

    def learn(
        self, t: int, context: Context, action: int, reward: float, next_context: Context, done: bool
    ) -> None:
        model_at_action_time, log = self._pending
        x = Transition(t, context, action, reward, next_context, done)
        self._buffer.append(x)
        self.q.update(x, model_at_action_time)
        if not log.revised:
            return
        new = self.cascade.incumbent
        contexts = [b.context for b in self._buffer]
        if new not in self.q.weights:
            # Start from the policy of the known model that predicts most like m'
            # over the window (the incumbent, when the revision is small).
            nearest = min(self.q.weights, key=lambda m: model_divergence(m, new, contexts))
            self.q.ensure(new, init_from=nearest if self._warm_start == "nearest" else log.incumbent_before)
        if not self._replay:
            self._buffer.clear()
            return
        # Algorithm 1 line 20: relabel the window with m' and replay it, step size
        # proportional to d(m_old, m') measured over the window's contexts.
        d = model_divergence(log.incumbent_before, new, contexts)
        lr = self.q.lr * d
        for _ in range(self._replay_epochs):
            self._buffer.relabel_and_replay(new, lambda tr, m: self.q.update(tr, m, lr=lr))
        self.replay_updates += self._replay_epochs * len(self._buffer)

    def predictive(self, context: Context) -> np.ndarray:
        return self.cascade.incumbent.predict_proba(context)

    def map_model(self) -> WeightedPreyModel:
        return self.cascade.incumbent


def make_learned_hh_predator(
    rng: np.random.Generator,
    kappa: float,
    replay: bool,
    tau: float = 1.0,
    delta: float = 0.1,
    window: int = 20,
    max_learned: int = 20,
    epsilon: float = 0.05,
    learning_rate: float = 0.3,
    gamma: float = 1.0,
    replay_epochs: int = 20,
    warm_start: str = "nearest",
) -> LearnedHHPredator:
    cascade = HHCascade(
        incumbent=DEN_A_MODEL,
        library=ModelLibrary(LIBRARY_0, max_learned=max_learned),
        kappa=kappa,
        window=window,
        verifier=HIMVerifier(tau=tau, delta=delta, contexts=None),
        revise=revise_weight_mixture,
        construct=construct_weight_model,
        merge_includes_incumbent=True,
    )
    return LearnedHHPredator(
        cascade, rng, replay, epsilon, learning_rate, gamma, replay_epochs, warm_start
    )
