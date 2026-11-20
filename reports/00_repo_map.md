# 00 — Repo map (Phase 0, orientation only)

Branch at time of mapping: `main` @ `87be574`. No code was changed in this phase.

## Read this first: three places where the code contradicts the brief

1. **This is not a JAX codebase.** It is plain Python + NumPy (+ pandas for aggregation). `pyproject.toml` lists only `numpy`; `jax` is not installed (`ModuleNotFoundError`). Agents are stateful Python objects stepped one round at a time by a Python loop (`himher/experiment/runner.py:28-38`). Rule 5 of the brief (jit / `lax.scan` / static shapes) therefore does not apply. All new code follows the existing object-per-agent style, so results stay exactly reproducible. Runs are parallelised across seeds with `multiprocessing` instead of `vmap`.
2. **Test count is 150, not 51.** `pytest` collects 150 tests. 143 are committed and 7 are in the untracked `tests/test_bocd_bayes.py` (the BOCD baseline, `himher/baselines/bocd_bayes.py`, is untracked too). All 150 pass. I have left both untracked files alone and have not committed them.
3. **"merge" is called `REVISE` in the code** (`himher/core/regime.py:27-30`, `Regime.REVISE = "revise"`). It is implemented as a θ-space interpolation between two library models (`himher/domains/handoff/revise.py`). In the new logs I use the brief's word `merge` and record the mapping here.

Smaller items:
- `pyarrow` isn't installed, so `per_step` is written as `.npz` (the brief allows this).
- `results/` is in `.gitignore` (`/results/`), as is `AAMAS 2027/`. New outputs go to `results/rebuttal_fri/`, which is therefore **not committed**. Reports under `reports/` are committed.
- The scripts that produced `AAMAS 2027/results/fig_*.csv`, `kld_raw.csv`, `kld_convergence_raw.csv` and `belief_true_type_*.csv` are **not in the repo**. Only the R notebook that reads them is (`AAMAS 2027/AAMAS 2027 plots.Rmd`). I reverse-checked the KLD definition against `model_trace_raw.csv` (see the metrics section below).

## Where things live

| Component | File : function / lines |
|---|---|
| Switching Handoff Game (env) | `himher/domains/handoff/env.py` — `SwitchingHandoffGame` (76-102), `SwitchSchedule` (44-73). Context `x ~ Bernoulli(0.5)` from the game RNG; counterpart samples its hand; reward = 1 iff the hands match. |
| Counterpart model m_θ = (θ₀, θ₁) | `himher/domains/handoff/counterpart.py` — `BernoulliHandModel`. `log_prob` clips to [1e-12, 1−1e-12]. |
| Library L₀ | `himher/experiment/conditions.py:16-19` — `M_L=(0.1,0.1)`, `M_R=(0.9,0.9)`, `M_C=(0.1,0.9)`; `LIBRARY_0=(M_L,M_R,M_C)`. New types: `M_NEW_1=(0.1,0.5)` (a merge of M_L and M_C), `M_NEW_2=(0.2,0.8)` (needs construct). |
| Library container | `himher/core/library.py` — `ModelLibrary`: the core L₀ is never evicted; up to `max_learned=20` learned entries (displaced incumbents only, see the Task E audit). `best_fit` = argmax window log-likelihood. |
| Conditions | `himher/experiment/conditions.py` — `stationary` (M_L throughout), `within_library_switch` (M_L→M_R at T/2), `outside_library_switch(M_NEW_1)` = **"m1"**, `outside_library_switch(M_NEW_2)` = **"m2"**, `recurring(M_NEW_2)` (M_L→m2→M_R→m2 at T/4 intervals). Run labels are in `scripts/run_experiments.py:158-170`. |
| Detection | `himher/core/detector.py` — `surprise_statistic` (mean −log p over the window, partial window before it fills) and `mismatch_detected` (`D_t > κ`, strict). Called in `himher/core/cascade.py:85-86`. |
| VOC check | `himher/core/voc.py` — `AlwaysProceedVOC` (always returns True). **No real VOC check exists**, so `n_voc_declined` is always 0. |
| Reuse / merge(=revise) / construct operators | Reuse: `ModelLibrary.best_fit` (`library.py:58-65`), called at `regime.py:65`. Merge: `revise_mixture_model` (`domains/handoff/revise.py:39-62`), a grid search over every library pair × p ∈ {0, 0.05, …, 1}. Construct: `construct_bernoulli_model` (`domains/handoff/construct.py`), a Beta(1,1) posterior mean over the window. |
| Regime ordering | `himher/core/regime.py:51-95` — `select_regime`: reuse → merge → construct, stopping at the first accepted candidate (`best_of=False` in all paper runs). |
| HIM acceptance test | `himher/core/him.py:64-95` — `HIMVerifier.evaluate`. Comparison pool built at `regime.py:62-63`. Audited in `reports/E_def2_audit.md`. |
| Cascade orchestration | `himher/core/cascade.py:78-130` — `HHCascade.step`. |
| HER relabel + replay buffer | `himher/core/her.py` — `ReplayBuffer` (deque with `maxlen`), `relabel_and_replay`. The buffer is created with `maxlen=cascade.window=20` (`baselines/hh_agent.py:38`). |
| Ego policy + update | `himher/domains/handoff/policy.py` — `TabularQPolicy`: Q(x, m, a) keyed by the model object; ε-greedy `act`; `update` is `Q += lr·(r − Q)`. |
| HH-R vs HH-A switch | `himher/baselines/hh_agent.py:28,74-81` — `replay=True` (HH-R) replays the buffer under the new incumbent on acceptance; `replay=False` (HH-A) clears it. |
| FB (fixed-library Bayes) | `himher/baselines/fixed_library_bayes.py` — categorical posterior over L₀; greedy posterior-predictive best response. |
| OB (oracle Bayes) | `himher/baselines/oracle_bayes.py` — FB with the condition's *last* true model added to the hypotheses. |
| CB (continuous Bayes) | `himher/baselines/continuous_bayes.py` — per-context Beta(α,α) (α=1) posterior mean over the **entire** history; greedy, no exploration; tie-break → L (`θ > 0.5` ⇒ R). |
| BOCD baseline (untracked, not in the paper) | `himher/baselines/bocd_bayes.py` |
| Runner | `himher/experiment/runner.py` — `run_episode(agent, condition, seed)`: game RNG = `default_rng(seed)`. The HH agent gets its own `default_rng(seed)` (a separate generator), so the env stream is identical across methods. |
| Config system | No config files. Constants are hard-coded at the top of each script (`scripts/run_experiments.py:49-92`, duplicated in `compute_phase_returns.py:42-51` and `compute_model_traces.py:184-193`). |
| Metrics | `himher/experiment/metrics.py` |
| Aggregation | `himher/experiment/aggregate.py` (mean ± SEM, LaTeX) |
| Figures | `AAMAS 2027/AAMAS 2027 plots.Rmd` (ggplot) reads `AAMAS 2027/results/fig_*.csv`: return before/after (Fig. 4), post-switch regret (Fig. 5), post-switch KLD (Fig. 6), KLD convergence, regime breakdown, policy updates, replay benefit, model accuracy. `scripts/make_figures.py` is an older matplotlib version. |

## Frozen hyperparameters (all match D6)

`κ = 0.6705027574263254` (Pass 1 sweep output; hard-coded in `compute_phase_returns.py:51`), `τ = 1.0`, `δ = 0.1`, `ε = 0.15`, `k = WINDOW = 20`, `MAX_LEARNED = 20`, `α = 1.0`, `LEARNING_RATE = 0.3`, `T = 300`. Note that `TabularQPolicy`'s *default* ε is 0.05, but every paper script passes 0.15. `scripts/run_variants.py` uses τ=0.5; that script is not part of the paper results.

## Reproducing the paper's results

```bash
python scripts/run_experiments.py   --seeds 200 --out "AAMAS 2027/results"   # kappa sweep + main_raw/main_summary/tables.tex
python scripts/compute_phase_returns.py --seeds 200 --out "AAMAS 2027/results" # phase_return_raw (Fig. 4)
python scripts/compute_model_traces.py  --seeds 200 --out "AAMAS 2027/results" # model_trace_raw / belief_trace_raw (→ KLD)
# fig_*.csv, kld_raw.csv, kld_convergence_raw.csv: generator not in repo
# then knit "AAMAS 2027/AAMAS 2027 plots.Rmd"
```
Seeds are `range(200)` = 0…199 for every condition × method.

## Episode structure and metric definitions

- `T = 300` rounds for every condition.
- First switch: Stationary none, Within-library 150, m1 150, m2 150, Recurring 75 (with further switches at 150 and 225).
- **pre/post-switch** (`metrics.phase_average_return`): "pre" = mean reward over rounds `t < t_switch`; "post" = mean over `t ≥ t_switch`, using the *first* switch only. For Stationary the existing phase script sets `t_switch = T`, so the whole episode is "pre" and "post" is NaN.
- **post_switch_regret** (`metrics.post_switch_regret`): Σ over t ≥ first switch of [max(θ_x, 1−θ_x) under the true model at t] − realised reward. It is computed against the *expected* optimal reward, not the realised oracle action, so it can be negative on individual rounds. For Recurring it covers all three later segments.
- **recovery time** = `metrics.adaptation_delay`: the first `t − t_switch` at which the trailing 20-round mean reward is ≥ 0.9 × the new model's steady-state return (mean over contexts of max(θ_x, 1−θ_x)). The earliest possible value is 19. NaN if never reached.
- **KLD** (Fig. 6): verified to equal ½·Σ_{x∈{0,1}} KL(Bern(θ^true_x) ‖ Bern(θ̂_x)) for the model held *after* the round-t update (max abs diff vs `kld_raw.csv` = 4e-16). The paper reports it only for HH-R, HH-A and CB.
- **False alarm** (paper): `false_alarm_rate` = detections / rounds, reported on Stationary.

## Timing (this machine, before any changes)

HH-R/HH-A ≈ 0.071 s per episode; FB/OB ≈ 0.005 s; CB ≈ 0.006 s (single core). With 16 logical cores, the full 200-seed matrix takes minutes, not hours.
