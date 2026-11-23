# HIM+HER in a reactive domain: Predator–Prey

**Why a second domain.** Reviewers noted that in the Switching Handoff Game
the counterpart never reacts to the ego: the surgeon's hand doesn't depend on
the robot. In Predator–Prey the prey's moves depend on where the predator
stands. Everything else is kept as close to the handoff setup as possible.

**Reproduce** (about 2 minutes on 10 cores; output in `results/predator_prey/`):

```
PYTHONPATH=. uv run --with numpy --with pandas python scripts/run_pp_experiments.py --seeds 200 --games 40
```

The code is in `himher/domains/predator_prey/`.

---

## 1. Domain

**Board and game.**
- A predator (ego *i*) chases a prey (counterpart *j*) on a 7×7 grid with dens
  at the top-left (A) and bottom-right (B).
- **Round:** both agents move at the same time. The prey moves N/S/E/W; the
  predator moves N/S/E/W or stays. A wall leaves an agent in place.
- **Context** x_t = (predator cell, prey cell).
- **Game:** up to T = 20 rounds. The prey starts in the central 3×3 and the
  predator at least 3 cells away (l1). The game ends when:
  - the predator **catches** the prey, by sharing a cell or swapping cells
    (+1);
  - the prey **enters either den** (0);
  - 20 rounds pass (**timeout**, 1/(1 + l1 distance)).

  Every round also costs −0.01.
- **Why stay and swap-catches:** without them the l1 distance keeps its
  parity, and about half the games could never end in a catch.
- **Episode:** 40 games. The predator keeps its library, models and Q-values
  across all of them. The prey's type switches **only between games**, at
  fixed games.

**Prey (ground truth and models share one form).**
- A type is w = (w_A, w_B, w_pred). Each of the prey's 4 moves gets three
  features in {−1, 0, +1}:
  - **f_A:** closer to / no change / further from den A, by shortest path with
    the predator's cell as a wall;
  - **f_B:** the same for den B;
  - **f_pred:** further from / no change / closer to the predator, by l1.
- The prey samples P(move | x) ∝ exp(w · f). It looks one move ahead and
  reacts to where the predator is now.
- At weight 1, a prey picks a good move about 88% of the time, the analogue of
  the surgeon's 0.1/0.9.

| Type | w | Behavior | Handoff analogue |
|---|---|---|---|
| Den A | (1, 0, 0) | runs for den A, around the predator | m_L (library, initial incumbent) |
| Den B | (0, 1, 0) | runs for den B | m_R (library) |
| Evade | (0, 0, 0.5) | flees the predator, ignores dens | m₂: held out, needs Construct |
| Cautious-B | (0, 0.5, 0.25) | heads for den B, keeps clear of the predator | m₁: midpoint of Den B and Evade, needs Merge |

**Conditions** (200 episodes each):

| Condition | Types in order | Switch at game | Handoff equivalent |
|---|---|---|---|
| Stationary | Den A | none | m_L |
| Within-library switch | Den A → Den B | 20 | m_L → m_R at round 150 |
| Outside-library switch (construct) | Den A → Evade | 20 | m_L → m₂ at round 150 |
| Outside-library switch (merge) | Den A → Evade → Cautious-B | 10, 20 | m_L → m₁ at round 150 |
| Recurring | Den A → Evade → Den B → Evade | 10, 20, 30 | rounds 75, 150, 225 |

- **Merge condition:** it needs an Evade phase first. Den A + Den B blends
  are useless, because the dens are in opposite corners and the pulls cancel.
  The useful blend needs Evade, which must first be built.
- **Design choices to disclose:**
  - Evade is half strength. At (0, 0, 1), no learned linear response could
    chase it (Section 3.5 reports that version too).
  - Either den ends the game, so the end condition doesn't depend on the
    hidden type.
  - The catch rate against den-runners is about 79%, above our 30–60% target.
    Changing the start distance or start rule didn't lower it.

---

## 2. Model (H+H instantiation)

**Cascade**

- **Detection** (as in the handoff game): D_t is the mean of −log P(a^j | x; m)
  over the last k = 20 prey moves. The window runs across game boundaries.
  Mismatch is flagged when D_t > κ. **κ = 1.3**, the smallest value with ≤ 5%
  false alarms in the Stationary condition (4.2%). It is higher than in the
  handoff game (0.67), because a guess among 4 moves is naturally more
  surprising than one between 2 hands.
- **Reuse:** the library model with the highest window likelihood.
- **Merge:** for each pair, blend (1 − p)·w_a + p·w_b, with p on a 21-point
  grid, and keep the best fit.
  - The pool is the library **plus the incumbent**, as in Algorithm 1's
    Merge(m, m̃).
  - Blending w combines **goals**: the blend weighs both in every round. It is
    not a mixture of move probabilities.
- **Construct:** a grid search over w ∈ {0, 0.5, …, 3}³ (343 candidates),
  keeping the best window likelihood. The handoff count estimate doesn't
  transfer, because positions rarely repeat. The output is an exact grid value,
  so a rebuilt model equals the stored one (needed for Reuse in Recurring).
- **HIM** (Definition 2): the candidate must beat {incumbent} ∪ L_t by τ = 1
  in log-likelihood, and differ from each by δ ≥ 0.1. The difference is total
  variation between predicted move probabilities, averaged over the window's
  positions, since 2,401 positions are too many to enumerate.

**Policy and HER** (HH-R)

- **Q is conditioned on the model, as in Section 5.2:**
  Q(x, m, a) = w_m · φ(t, x, a), with one 4-weight vector per model.
  - A table, as in the handoff game, is impossible with 2,401 positions.
  - φ = [1, Q*_A, Q*_B, Q*_W]: the planned values of move a against Den A,
    against Den B, and against a worst-case prey.
  - These three are computed once, from the game rules and the starting
    library only. No held-out type is used.
  - Q for any model is a learned weighting of these values: transfer from
    known types, in the spirit of generalized policy improvement.
- **Initialization:**
  - Den A and Den B start with their exact responses, (0, 1, 0, 0) and
    (0, 0, 1, 0).
  - A new model starts from the weights of the known model whose predictions
    are closest to it (Section 4.3: start from the previous policy).
- **Online learning:**
  - Q-learning on each round, with target r + max_a′ Q(next state, m, a′)
    and learning rate 0.3;
  - ε-greedy with ε = 0.05.
- **HER** (Algorithm 1, line 20): on acceptance of m′, relabel the 20-round
  window with m′ and replay it 20 times, with step size 0.3 · d(m, m′).
- **HH-A** (ablation): the same, but the window is discarded instead of
  replayed.

**Comparison methods**
- **HH-plan:** the same cascade and models as HH-R, but it acts by exact
  planning against its current model. There is no policy to learn, so HER
  doesn't apply. It isolates the cascade.
- **Fixed Bayes:** a posterior over {Den A, Den B}.
- **Oracle Bayes:** a posterior that also includes the condition's held-out
  types; an upper bound, as in the paper.
- **Continuous Bayes:** the best w on the 343-point grid, refit on the
  **entire** history every round.
- **Oracle:** knows the true type of every game.

As in the paper, the baselines act on their beliefs. Here "acting" means
exact finite-horizon value iteration over all positions; Oracle Bayes acts on
posterior-weighted Q-values.

**Built-in knowledge.** H+H gets:
- the prey's model form;
- the game rules (ω);
- the starting library and its responses;
- the three planned-value features.

The baselines get the same or more: they plan exactly against any model, and
Oracle Bayes is told the held-out types. The features and the half-strength
Evade were chosen after simpler versions failed (Section 3.5).

| Parameter | Value | How it was set |
|---|---|---|
| k (window), τ, δ | 20, 1.0, 0.1 | same as the handoff game |
| κ | 1.3 | false-alarm sweep (≤ 5%), 200 seeds |
| ε, learning rate, replay passes | 0.05, 0.3, 20 | small sweep on separate seeds (2000+) |

---

## 3. Results

200 episodes per condition and method. Standard errors are about 0.005–0.010
for reward.

### 3.1 Reward per game after the first switch

| Condition | HH-R | HH-A | HH-plan | Fixed Bayes | Oracle Bayes† | Cont. Bayes | Oracle† |
|---|---|---|---|---|---|---|---|
| Within-library | **0.637** | 0.557 | 0.712 | 0.438 | 0.438 | 0.538 | 0.743 |
| Outside (construct) | 0.684 | 0.648 | 0.760 | 0.676 | 0.759 | **0.742** | 0.798 |
| Outside (merge) | 0.717 | 0.646 | 0.797 | 0.678 | 0.763 | **0.745** | 0.824 |
| Recurring | 0.670 | 0.617 | 0.750 | 0.622 | 0.734 | **0.715** | 0.780 |

† Upper bounds. Bold marks the best of HH-R, HH-A, Fixed Bayes and Continuous
Bayes.

- **Before the switch:** HH-R and HH-A earn 0.69–0.70 per game, and the
  planning methods 0.73–0.74. This is the cost of learning, as in Fig. 4 of
  the paper.
- **Stationary:** HH-R 0.699, Fixed Bayes 0.738.

### 3.2 Model accuracy after the last switch: KL(m_true ‖ m)

| Condition | HH-R | Fixed Bayes | Oracle Bayes† | Cont. Bayes |
|---|---|---|---|---|
| Stationary | 0.046 | 0.003 | 0.003 | 0.044 |
| Within-library | **0.169** | 1.102 | 1.102 | 0.582 |
| Outside (construct) | **0.073** | 0.530 | 0.200 | 0.187 |
| Outside (merge) | **0.054** | 0.498 | 0.120 | 0.147 |
| Recurring | 0.089 | 0.483 | 0.000 | **0.016** |

l2 distance in w is not a good accuracy measure here. In open ground
f_B ≈ −f_A, so different w's predict nearly the same moves.

### 3.3 Detection, regimes and policy

- **Detection:** the false-alarm rate is 4–7%. Detection takes about 3.5 moves
  after a den change and about 7 after Evade appears.
- **Revisions per episode:** about 2 in the Stationary condition (usually to a
  slightly noisier Den A) and 5–12 with switches.
- **Regime that first reached the exact true w** (HH-R, out of 200 episodes):

  | Target | First via | Episodes |
  |---|---|---|
  | Den B | Reuse | 66 |
  | Evade | Construct | 84–89 |
  | Stored Evade, at game 30 (Recurring) | Reuse | 83 |
  | Cautious-B | Merge | 8 |

  Reuse of Den B usually pre-empts Merge, because Cautious-B is close to
  Den B.
- **Catch rates** (over switch-condition episodes):
  - HH-R 73–78%, HH-A 69–74%, baselines 65–83%;
  - rounds to catch are about 6 against den-runners and about 7.5–8 once
    Evade appears.
- **Agreement with the oracle's best move:** HH-R 65–80%, HH-A 58–79%.

### 3.4 Interpretation

1. **HER helps in every switch condition.** HH-R beats HH-A by 0.080
   (within), 0.036 (construct), 0.071 (merge) and 0.053 (Recurring). A new
   model starts from a related model's response; replaying the window HIM
   just verified retrains it at once. In the handoff game, replay lost on the
   within-library switch.
2. **H+H has the most accurate model** after every switch except Recurring,
   where Oracle Bayes was given the type and Continuous Bayes has seen Evade
   twice.
3. **H+H wins on reward within the library**, against all baselines. Fixed
   Bayes and Oracle Bayes stay anchored on 20 games of Den A evidence, and
   Continuous Bayes fits a blend of old and new behavior.
4. **Outside the library, Continuous Bayes earns more reward despite a worse
   model.**
   - It plans its response exactly, while H+H must learn a response to a model
     it just built.
   - Against a noisy evader, a rough model with an exact response beats an
     accurate model with a partly learned one.
   - HH-plan, the same cascade with exact planning, beats Continuous Bayes
     (0.75–0.80). So the gap is the cost of learning the response, not of
     modeling.
   - In the handoff game the response was a 4-entry table learned in a few
     rounds, so this cost was negligible.
5. **Cost isn't measured,** in the paper or here. HH-R never plans after the
   start, while Continuous Bayes re-plans whenever its best-fit model changes,
   so H+H is likely much cheaper. This still needs to be counted.

### 3.5 Sensitivity: full-strength Evade

Evade = (0, 0, 1) and Cautious-B = (0, 0.5, 0.5), with the same settings and
seeds. Reward per game after the switch:

| Condition | HH-R | HH-A | HH-plan | Fixed Bayes | Oracle Bayes† | Cont. Bayes |
|---|---|---|---|---|---|---|
| Outside (construct) | 0.406 | 0.357 | 0.632 | 0.388 | 0.558 | **0.548** |
| Outside (merge) | 0.549 | 0.482 | 0.712 | 0.534 | 0.665 | **0.640** |
| Recurring | 0.479 | 0.433 | 0.645 | 0.442 | 0.615 | **0.610** |

- HER still helps (+0.05–0.07), and HH-R still beats Fixed Bayes.
- The gap to Continuous Bayes widens to 0.09–0.14: a full-strength evader is
  hard to chase with a learned linear response.
- Even the best-fitting weights reach only 0.52 per game, against the
  oracle's 0.69.
- HH-plan again beats every baseline.

### 3.6 Open items

- **Measure computational cost:** plans computed, likelihood evaluations and
  wall-clock time per method.
- **The merge test is weak:** Cautious-B is too close to Den B. A blend
  further from both parents would make a cleaner test.
- **The catch rate against den-runners is about 79%**; lowering it needs a
  rule change, such as a bigger grid.
- **Handoff paper:** Section 5.3 gives m₂ = (0.8, 0.2), but the code uses
  (0.2, 0.8).
