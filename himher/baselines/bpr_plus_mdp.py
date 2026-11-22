"""BPR+ with MDP opponent models (rebuttal, 11/22), following
Hernandez-Leal, Taylor, Rosman, Sucar & Munoz de Cote, "Identifying and
Tracking Switching, Non-Stationary Opponents: A Bayesian Approach",
AAAI-16 Workshop WS-16-11, sections "Learning opponent strategies",
"Updating performance models", Algorithms 1-2.

Paper element -> this game
--------------------------
* Opponent strategy model: an MDP M = <S, A, T, R> over attributes the
  opponent conditions on. The handoff counterpart conditions on the
  context only, so S = {x = 0, x = 1}; A = {L, R} (the ego's actions).
  - Known strategies (L0): their MDPs are given (the paper assumes them):
    R(s, R) = theta_s, R(s, L) = 1 - theta_s (expected reward = P(match)),
    T(s, a, s') = 0.5 (contexts are i.i.d. fair coins, independent of a).
  - New strategies: learned from the exploratory phase by counts, exactly
    as in the paper: T_hat(s,a,s') = n(s,a,s') / m(s,a),
    R_hat(s,a) = r(s,a) / m(s,a). (s,a) pairs never visited in the phase
    get R_hat = 0.5 and a uniform T_hat (the paper does not say).
* Optimal policy: value iteration on the MDP (gamma = 0.9, stop when
  max |dV| < 1e-10); greedy in Q, ties -> L.
* Performance model P(U | tau, pi): the paper simulates pi on M_tau and
  fits a Gaussian to the rewards. Simulating a stochastic-reward MDP gives
  0/1 rewards with per-state mean R(s, pi(s)); we compute the Gaussian's
  moments exactly (equivalent to an infinitely long simulation, and no
  RNG): mu = sum_s d_pi(s) R(s, pi(s)) with d_pi the stationary state
  distribution under pi, sigma^2 = mu (1 - mu) (a mixture of Bernoullis is
  Bernoulli(mu)), floored at sigma_min = 0.05.
* Signal: the immediate reward of the round (paper's choice). Belief
  update: Bayes with the Gaussian density N(r; mu_{tau,pi_t}, sigma),
  floored at ``belief_floor`` then renormalised (paper: "restrict all
  beliefs to not drop to zero").
* Policy selection: probability of improvement,
  argmax_pi sum_tau beta(tau) P(U > U+ | tau, pi) with Gaussian P and
  U+ = max_pi sum_tau beta(tau) mu_{tau,pi} + xi (xi = 0.01; the paper only
  says U+ > U_hat). Re-selected every round.
* Drift exploration: epsilon-greedy over *policies* (with probability
  epsilon a uniformly random library policy is executed). The reward is a
  signal for the executed policy's performance model, so the belief update
  stays well-defined on exploration rounds.
* New-model detection (paper): p_i(tau) = P(r_i | tau, pi_i) from the
  performance model (a density here); p_hat = mean over the last n
  rounds; if p_hat(tau) < rho for every tau, start learning.
* Learning phase: ``learn_rounds`` rounds of uniformly random actions
  (paper: "for example, with random actions"); no belief updates. Then the
  MDP is learned and solved, its policy joins the library, performance
  models for every (tau, pi) pair are (re)computed (groups 2-4), and the
  belief puts all mass on the new model except ``belief_floor`` elsewhere.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand

S, A = 2, 2  # contexts x in {0, 1}; actions L = 0, R = 1
SIGMA_MIN = 0.05
GAMMA = 0.9
VI_TOL = 1e-10
XI = 0.01


@dataclass
class ContextMDP:
    T: np.ndarray  # (S, A, S)
    R: np.ndarray  # (S, A) expected reward
    implied_theta: tuple[float, float]  # P(counterpart plays R | x), for logging only

    @classmethod
    def from_model(cls, m: BernoulliHandModel) -> "ContextMDP":
        R = np.array([[1.0 - m.theta0, m.theta0], [1.0 - m.theta1, m.theta1]])
        return cls(np.full((S, A, S), 0.5), R, m.theta)

    @classmethod
    def from_counts(cls, steps: list[tuple[int, int, float]]) -> "ContextMDP":
        """steps: (context, action, reward) in time order; transitions are
        consecutive contexts within the learning phase."""
        n_sas = np.zeros((S, A, S))
        r_sum = np.zeros((S, A))
        m_sa = np.zeros((S, A))
        for i, (s, a, r) in enumerate(steps):
            r_sum[s, a] += r
            m_sa[s, a] += 1
            if i + 1 < len(steps):
                n_sas[s, a, steps[i + 1][0]] += 1
        R = np.where(m_sa > 0, r_sum / np.maximum(m_sa, 1), 0.5)
        n_sa = n_sas.sum(axis=2, keepdims=True)
        T = np.where(n_sa > 0, n_sas / np.maximum(n_sa, 1), 1.0 / S)
        theta = []
        for s in range(S):
            m = m_sa[s].sum()
            # MLE of P(counterpart R | s): matches when playing R, misses when playing L.
            theta.append(float((r_sum[s, 1] + (m_sa[s, 0] - r_sum[s, 0])) / m) if m > 0 else 0.5)
        return cls(T, R, (theta[0], theta[1]))


def value_iteration(mdp: ContextMDP, gamma: float = GAMMA, tol: float = VI_TOL) -> tuple[np.ndarray, int]:
    """Returns (policy[s] in {0, 1}, number of sweeps)."""
    V = np.zeros(S)
    sweeps = 0
    while True:
        sweeps += 1
        Q = mdp.R + gamma * mdp.T @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = mdp.R + gamma * mdp.T @ V
    policy = np.where(Q[:, 1] > Q[:, 0], 1, 0)  # ties -> L
    return policy, sweeps


def stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a 2-state chain P[s, s']."""
    a, b = P[0, 1], P[1, 0]
    if a + b == 0:
        return np.full(S, 1.0 / S)
    return np.array([b / (a + b), a / (a + b)])


def performance_model(mdp: ContextMDP, policy: np.ndarray) -> tuple[float, float]:
    P = np.array([mdp.T[s, policy[s]] for s in range(S)])
    d = stationary(P)
    mu = float(sum(d[s] * mdp.R[s, policy[s]] for s in range(S)))
    return mu, max(SIGMA_MIN, math.sqrt(max(mu * (1.0 - mu), 0.0)))


def gauss_pdf(r: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((r - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def policy_scores(belief: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability of improvement per policy; mu, sigma are (|O|, |Pi|)."""
    expected = belief @ mu
    u_plus = float(expected.max()) + XI
    z = (u_plus - mu) / sigma
    p_improve = 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))  # P(U > U+)
    return belief @ p_improve


class BPRPlusMDPAgent:
    def __init__(
        self,
        library: tuple[BernoulliHandModel, ...],
        rng: np.random.Generator,
        rho: float,
        n: int = 7,
        epsilon: float = 0.15,
        learn_rounds: int = 20,
        belief_floor: float = 0.01,
    ) -> None:
        if not library:
            raise ValueError("library must be non-empty")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if n <= 0 or learn_rounds <= 0:
            raise ValueError("n and learn_rounds must be positive")
        if not 0.0 < belief_floor < 1.0 / len(library):
            raise ValueError("belief_floor must be in (0, 1/|library|)")
        self._rng = rng
        self._rho, self._n, self._epsilon = rho, n, epsilon
        self._learn_rounds, self._floor = learn_rounds, belief_floor
        self._mdps: list[ContextMDP] = []
        self._policies: list[np.ndarray] = []
        self.vi_sweeps: list[int] = []
        for m in library:
            self._add_mdp(ContextMDP.from_model(m))
        self._belief = np.full(len(self._mdps), 1.0 / len(self._mdps))
        self._recent: deque[np.ndarray] = deque(maxlen=n)
        self._learning: list[tuple[int, int, float]] | None = None
        self._current_policy = 0
        self._t = 0
        self.detection_times: list[int] = []
        self.learned_models: list[tuple[int, BernoulliHandModel]] = []

    # -- library ------------------------------------------------------------
    def _add_mdp(self, mdp: ContextMDP) -> None:
        policy, sweeps = value_iteration(mdp)
        self.vi_sweeps.append(sweeps)
        self._mdps.append(mdp)
        self._policies.append(policy)
        k = len(self._mdps)
        self._mu = np.zeros((k, k))
        self._sigma = np.zeros((k, k))
        for i, M in enumerate(self._mdps):
            for j, pi in enumerate(self._policies):
                self._mu[i, j], self._sigma[i, j] = performance_model(M, pi)

    # -- read-only views (instrumentation.held_theta uses posterior/hypotheses)
    @property
    def hypotheses(self) -> tuple[BernoulliHandModel, ...]:
        return tuple(BernoulliHandModel(*m.implied_theta) for m in self._mdps)

    @property
    def posterior(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def learning(self) -> bool:
        return self._learning is not None

    @property
    def policies(self) -> tuple[np.ndarray, ...]:
        return tuple(p.copy() for p in self._policies)

    def _pi_choice(self) -> int:
        return int(np.argmax(policy_scores(self._belief, self._mu, self._sigma)))

    def greedy_action(self, context: Context) -> Hand:
        return Hand(int(self._policies[self._pi_choice()][context]))

    # -- Agent interface ----------------------------------------------------
    def act(self, context: Context) -> Hand:
        if self._learning is not None:
            return Hand(int(self._rng.integers(0, 2)))
        if self._rng.random() < self._epsilon:
            self._current_policy = int(self._rng.integers(0, len(self._policies)))
        else:
            self._current_policy = self._pi_choice()
        return Hand(int(self._policies[self._current_policy][context]))

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        t = self._t
        self._t += 1
        if self._learning is not None:
            self._learning.append((int(context), int(ego_action), float(reward)))
            if len(self._learning) >= self._learn_rounds:
                self._finish_learning(t)
            return
        j = self._current_policy
        p = np.array([gauss_pdf(reward, self._mu[i, j], self._sigma[i, j]) for i in range(len(self._mdps))])
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
        mdp = ContextMDP.from_counts(self._learning)
        self._learning = None
        self._add_mdp(mdp)
        self.learned_models.append((t, BernoulliHandModel(*mdp.implied_theta)))
        k = len(self._mdps)
        belief = np.full(k, self._floor)
        belief[-1] = 1.0 - self._floor * (k - 1)
        self._belief = belief
        self._recent.clear()
