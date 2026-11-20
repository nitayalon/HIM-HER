# Task E — Audit of HIM acceptance against Definition 2 (report only; no code changed)

> **⚠ Headline for the author.** The code does **not** implement Definition 2 literally. It compares the candidate against {incumbent} ∪ lib_t **with the candidate itself removed**. The docstring says this is deliberate: it is "the only reading under which reuse can ever be accepted" (`himher/core/him.py:71-78`). The threshold is also **strict** (`Δ > τ`), not `≥`, and Δ is a **summed** (not per-step) window log-likelihood. Nothing was changed. Every reported result was produced by the code as described below.

The code calls this test "Definition 4.1"; the brief calls it "Definition 2". They are the same test. All line numbers refer to `main @ 87be574`.

## 1. Which models is the candidate compared against?

The same rule applies to all three branches. `select_regime` builds the pool, and the verifier prepends the incumbent:

```python
# himher/core/regime.py:62-63
def comparison_pool(candidate):
    return tuple(m for m in library if m != incumbent and m != candidate)

# himher/core/him.py:79-85
comparison_set = (incumbent, *comparison_pool)
candidate_score = score(candidate, window)
improvements   = [candidate_score - score(m, window) for m in comparison_set]
distinctnesses = [model_divergence(candidate, m, self._contexts) for m in comparison_set]
```

The comparison set is therefore **{incumbent} ∪ (lib_t \ {incumbent, candidate})**, i.e. {m} ∪ lib_t minus the candidate. Equality is dataclass value equality on (θ₀, θ₁).

| Branch | Candidate | Compared against | Extra pre-filter |
|---|---|---|---|
| reuse | `library.best_fit(window)`, the argmax log-likelihood over all of lib_t (`regime.py:65`) | incumbent + the rest of lib_t | Skipped entirely if the argmax *is* the incumbent (`regime.py:67`) |
| merge (`Regime.REVISE`) | best (pair, p) θ-interpolation over all library pairs, p on a 21-point grid (`revise.py:39-62`) | incumbent + lib_t (minus the candidate if it happens to equal a library entry) | Skipped if the candidate equals the incumbent (`regime.py:76`) |
| construct | Beta(1,1) posterior mean on the window (`construct.py`) | incumbent + lib_t | none |

What lib_t contains: L₀ = {M_L, M_R, M_C} (never evicted) plus up to `MAX_LEARNED=20` **displaced incumbents** (`cascade.py:113-115`). Rejected candidates are *not* added (`cascade.py:71-76,108-110`).

**Doc inconsistencies (no behavioural effect):**
- `library.py:1-6` says rejected candidates are retained. They are not.
- The comment at `scripts/run_experiments.py:56-60` says the verifier is "scoped to the incumbent only". The code compares against the full pool; `him.py:8-23` documents the reversal of that earlier scoping.

Consequence for reuse: the reuse candidate is the library argmax, so its improvement over every *other* library model is ≥ 0 automatically. To be accepted it must still beat **the runner-up library model** (not just the incumbent) by more than τ = 1 nat on the window.

## 2. In-sample or a separate window?

**In-sample.** One window, `self.history.as_tuple()` (`cascade.py:102`), is used to:
- select the reuse candidate (`best_fit`),
- fit the merge grid search,
- fit construct, and
- run the HIM test.

It is also the same window that the detector uses (`cascade.py:85`). It holds the last ≤ k = 20 observations *including the current round*. Before round 20 it is a partial window (`history.py`).

## 3. Is there a distinctness check?

Yes, condition (ii):

```python
# himher/core/model.py:57-69
def model_divergence(m, m_prime, contexts):
    distances = [total_variation_distance(m.predict_proba(c), m_prime.predict_proba(c)) for c in contexts]
    return float(np.mean(distances))
# him.py:88-89
min_distinctness = min(distinctnesses)
accepted = (min_improvement > self._tau) and (min_distinctness >= self._delta)
```

For Bernoulli hand models this is d(m, m') = ½(|θ₀−θ₀'| + |θ₁−θ₁'|), the mean over contexts {0,1} of the TV distance. The candidate must satisfy d ≥ δ = 0.1 against **every** model in the comparison set from item 1. The check uses `≥`.

There is also an exact-equality pre-filter, candidate ≠ incumbent, for reuse and merge (item 1 table).

## 4. Threshold form

- `min_improvement > τ`: **strict** (`him.py:89`).
- Δ is the **summed** log-likelihood over the window: `history_log_likelihood` = Σ_τ log p(a_τ | x_τ; m) (`model.py:39-48`), not a per-step average.
- τ = 1.0 nat summed over ≤ 20 observations (≈ 0.05 nat/step at a full window).
- `log_prob` clips probabilities to [1e-12, 1−1e-12] (`counterpart.py:49`).

Note on the reviewer's point: with a strict `>`, a self-comparison (Δ = 0) would fail even at τ = 0.

## 5. Reuse acceptances per condition

From the stored paper results (`AAMAS 2027/results/main_raw.csv`, `method == hh_replay`, 200 seeds each). These counts are identical for HH-A, since the cascade does not depend on ego actions.

| Condition | reuse accepted (total) | per episode | episodes with ≥1 reuse | merge accepted / ep | construct accepted / ep | detections / ep |
|---|---|---|---|---|---|---|
| stationary | 92 | 0.46 | 54 / 200 | 0.745 | 0.075 | 2.19 |
| within_library_switch | 166 | 0.83 | 80 / 200 | 2.20 | 1.03 | 9.16 |
| outside_library_switch_m1 | 340 | 1.70 | 139 / 200 | (see summary) | | 10.16 |
| outside_library_switch_m2 | 322 | 1.61 | 151 / 200 | | | 10.37 |
| recurring | 583 | 2.92 | 189 / 200 | | | 15.24 |

**Cross-check against the logged rerun (Task L, default flags, all 200 seeds):** see `reports/FRIDAY_SUMMARY.md` §2. The logged rerun reproduces these counts exactly. Each accepted reuse is also logged with its binding comparison model and Δ in `results/rebuttal_fri/main/events.csv`.

## 6. Conclusion

The code does not match Definition 2 as written. Read literally, Definition 2 would make every reuse fail, because a library model compared with itself gives Δ = 0. The code instead implements the following:

> *A candidate m′ (from reuse, merge or construct) is accepted iff, on the same length-k window h_{t−k:t} from which it was selected, its summed log-likelihood exceeds that of every other model in {m_t} ∪ lib_t (m′ itself excluded) by strictly more than τ, and its context-averaged total-variation distance from each of those models is at least δ; here lib_t is L₀ plus previously accepted (displaced) incumbents, not rejected candidates.*

Further points for the author to decide on:
- The test is **in-sample**: the same window is used for selection and verification.
- The code uses strict `>` for τ and `≥` for δ.
- Reuse skips the incumbent before HIM is consulted.

I have not fixed anything. Changing any of these would change every reported number.
