"""BPR+ baseline (rebuttal): Hernandez-Leal, Taylor, Rosman, Sucar & Munoz
de Cote, "Identifying and Tracking Switching, Non-Stationary Opponents: A
Bayesian Approach", AAAI-16 Workshop on Multiagent Interaction without
Prior Coordination (WS-16-11), Algorithms 1-2.

Mapping onto the Switching Handoff Game
---------------------------------------
* Opponent strategies O: Bernoulli hand models tau = (theta0, theta1);
  initially L0 = {M_L, M_R, M_C}, plus every model learned online.
* Policy library Pi: the optimal (best-response) policy against each
  strategy, pi_tau(x) = R iff theta_tau,x > 0.5 (ties -> L). The paper
  solves the opponent's induced MDP by value iteration; this game is
  one-shot per round, so value iteration reduces to this best response.
* Signal sigma: the immediate reward (the paper's choice). Given the ego's
  own action, r = 1 iff the counterpart matched, so the reward carries the
  same information as the counterpart's hand.
* Performance models P(U | tau, pi): the paper fits Gaussians to rewards
  simulated on the learned MDP. Rewards here are {0, 1}, so we use the
  exact distribution: Bernoulli with
  P(r = 1 | tau, a, x) = theta_tau,x if a == R else 1 - theta_tau,x.
* Policy selection: probability of improvement,
  argmax_pi sum_tau beta(tau) P(U > U+ | tau, pi) with U+ > U_hat. For
  U in {0, 1} and any U+ in (U_hat, 1), P(U > U+) = P(U = 1), so PI is
  exactly argmax expected reward under the belief (computed per round,
  in the current context). Ties go to the earliest library policy.
* Belief update: Bayes' rule on sigma, then "restrict all beliefs to not
  drop to zero by adding a small epsilon" -- implemented as a floor
  (``belief_floor``) followed by renormalisation.
* Drift exploration: epsilon-greedy (the paper names epsilon-greedy and
  softmax as valid choices; no value is given).
* New-model detection: p_i(tau) = P(r_i | tau, pi_i) for the round's
  executed action; p_hat(tau) = mean of p over the last n rounds. If
  p_hat(tau) < rho for *every* tau, a new strategy is declared.
* Learning a new strategy: an exploratory phase of ``learn_rounds``
  rounds with uniformly random actions (the paper: "for example, with
  random actions"), assuming no switch during it. The induced model is
  learned by counts; we add the Beta(alpha, alpha) smoothing used by the
  paper's other count models (construct_bernoulli_model) so a context
  unseen in the phase falls back to 0.5. The learned model and its best
  response join O and Pi; performance models for every (tau, pi) pair
  (the paper's groups 2-4) follow from the Bernoulli form. The paper
  does not say how the belief is set after learning; we put all mass on
  the new model except ``belief_floor`` on every other one.

Hyper-parameters not fixed by the paper (epsilon, learning-phase length,
belief floor, rho for this game) are set in
himher/experiment/rebuttal_methods.py; rho is calibrated there.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


def best_response(model: BernoulliHandModel, context: Context) -> Hand:
    theta_x = model.theta1 if context else model.theta0
    return Hand.R if theta_x > 0.5 else Hand.L


def reward_prob(model: BernoulliHandModel, context: Context, action: Hand, reward: float) -> float:
    """Performance model P(r | tau, a, x) for r in {0, 1}."""
    theta_x = model.theta1 if context else model.theta0
    p_match = theta_x if action == Hand.R else 1.0 - theta_x
    return p_match if reward >= 0.5 else 1.0 - p_match


class BPRPlusAgent:
    def __init__(
        self,
        library: tuple[BernoulliHandModel, ...],
        rng: np.random.Generator,
        rho: float,
        n: int = 7,
        epsilon: float = 0.15,
        learn_rounds: int = 20,
        belief_floor: float = 0.01,
        alpha: float = 1.0,
    ) -> None:
        if not library:
            raise ValueError("library must be non-empty")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if n <= 0 or learn_rounds <= 0:
            raise ValueError("n and learn_rounds must be positive")
        if not 0.0 < belief_floor < 1.0 / len(library):
            raise ValueError("belief_floor must be in (0, 1/|library|)")
        self._types: list[BernoulliHandModel] = list(library)
        self._belief = np.full(len(library), 1.0 / len(library))
        self._rng = rng
        self._rho = rho
        self._n = n
        self._epsilon = epsilon
        self._learn_rounds = learn_rounds
        self._floor = belief_floor
        self._alpha = alpha
        self._recent: deque[np.ndarray] = deque(maxlen=n)  # p_i(tau) per round
        self._learning: list[tuple[Context, Hand]] | None = None
        self._t = 0
        # Bookkeeping (read-only for logging).
        self.detection_times: list[int] = []
        self.learned_models: list[tuple[int, BernoulliHandModel]] = []
        self.last_greedy: Hand | None = None

    # -- read-only views (instrumentation.held_theta uses posterior/hypotheses)
    @property
    def hypotheses(self) -> tuple[BernoulliHandModel, ...]:
        return tuple(self._types)

    @property
    def posterior(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def learning(self) -> bool:
        return self._learning is not None

    def greedy_action(self, context: Context) -> Hand:
        """Probability-of-improvement choice = argmax expected reward over
        the library policies under the current belief."""
        best_a, best_u = None, -np.inf
        for pi_model in self._types:
            a = best_response(pi_model, context)
            u = sum(b * reward_prob(tau, context, a, 1.0) for b, tau in zip(self._belief, self._types))
            if u > best_u:
                best_a, best_u = a, u
        return best_a

    def act(self, context: Context) -> Hand:
        greedy = self.greedy_action(context)
        self.last_greedy = greedy
        if self._learning is not None:
            return Hand(int(self._rng.integers(0, 2)))
        if self._rng.random() < self._epsilon:
            return Hand(int(self._rng.integers(0, 2)))
        return greedy

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        t = self._t
        self._t += 1
        if self._learning is not None:
            self._learning.append((context, counterpart_action))
            if len(self._learning) >= self._learn_rounds:
                self._finish_learning(t)
            return

        p = np.array([reward_prob(tau, context, ego_action, reward) for tau in self._types])
        post = self._belief * p
        post = post / post.sum() if post.sum() > 0 else np.full_like(post, 1.0 / len(post))
        post = np.maximum(post, self._floor)
        self._belief = post / post.sum()

        self._recent.append(p)
        if len(self._recent) == self._n:
            p_hat = np.mean(np.stack(self._recent), axis=0)
            if np.all(p_hat < self._rho):
                self.detection_times.append(t)
                self._learning = []
                self._recent.clear()

    def _finish_learning(self, t: int) -> None:
        new_model = construct_bernoulli_model(self._learning, self._alpha)
        self._learning = None
        self._types.append(new_model)
        self.learned_models.append((t, new_model))
        k = len(self._types)
        belief = np.full(k, self._floor)
        belief[-1] = 1.0 - self._floor * (k - 1)
        self._belief = belief
        self._recent.clear()
