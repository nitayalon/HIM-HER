Serial run (1 worker, `log_level=none`, seeds 0–19 × 5 conditions = 100 runs per row). Hardware: Intel64 Family 6 Model 197 Stepping 2, GenuineIntel, 16 logical cores, Windows-11-10.0.26300-SP0; Python 3.14.7, NumPy 2.5.2; no JAX, so there is no compile time. Likelihood evaluations = calls to `BernoulliHandModel.log_prob` made by the agent itself. Logging's own re-scoring is excluded, and FB/OB evaluate every hypothesis each round.

| method | variant | wall/step mean (µs) | wall/step std (µs) | wall/run mean (ms) | loglik evals/run | revisions/run |
|---|---|---|---|---|---|---|
| HH-R | default | 333.1 | 2024.8 | 100.21 | 39,997 | 4.47 |
| HH-A | default | 327.7 | 2011.1 | 98.59 | 39,997 | 4.47 |
| FB | default | 15.2 | 7.4 | 4.83 | 900 | — |
| OB | default | 15.6 | 7.5 | 4.91 | 1,080 | — |
| CB | default | 18.3 | 11.5 | 5.69 | 0 | — |
| WB-k20 | default | 2.1 | 1.8 | 0.81 | 0 | — |
| DB-0.90 | default | 2.0 | 2.0 | 0.78 | 0 | — |
| DB-0.95 | default | 2.0 | 2.8 | 0.77 | 0 | — |
| DB-0.98 | default | 2.0 | 1.9 | 0.77 | 0 | — |
| HH-R | him_window | 326.3 | 2010.4 | 98.16 | 39,997 | 4.47 |
| HH-R | post_detection | 328.6 | 2015.1 | 98.87 | 39,997 | 4.47 |
| HH-R | decay_reset | 332.9 | 2039.2 | 100.15 | 39,997 | 4.47 |
| HH-R | him_window+decay_reset | 330.1 | 2023.2 | 99.32 | 39,997 | 4.47 |
