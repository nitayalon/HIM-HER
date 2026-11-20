
**HH-R (default) minus windowed / discounted Bayes.** Positive = A higher. Mean paired difference over 200 seeds [95% bootstrap CI]. `*` = CI excludes 0.

| condition | A − B | return_post | regret_post | auc |
|---|---|---|---|---|
| within_library_switch | HH-R − WB-k20 | -0.063 [-0.076, -0.051]* | 9.49 [7.60, 11.43]* | -39.19 [-41.97, -36.46]* |
| outside_library_switch_m1 | HH-R − WB-k20 | -0.032 [-0.037, -0.027]* | 4.78 [4.00, 5.57]* | -34.48 [-36.92, -32.10]* |
| outside_library_switch_m2 | HH-R − WB-k20 | -0.044 [-0.052, -0.037]* | 6.66 [5.57, 7.86]* | -36.35 [-38.79, -33.97]* |
| recurring | HH-R − WB-k20 | -0.048 [-0.056, -0.041]* | 10.82 [9.14, 12.56]* | -34.47 [-36.79, -32.21]* |
| within_library_switch | HH-R − DB-0.90 | -0.104 [-0.117, -0.091]* | 15.54 [13.67, 17.48]* | -44.95 [-47.73, -42.23]* |
| outside_library_switch_m1 | HH-R − DB-0.90 | -0.031 [-0.036, -0.025]* | 4.59 [3.81, 5.39]* | -34.00 [-36.45, -31.64]* |
| outside_library_switch_m2 | HH-R − DB-0.90 | -0.059 [-0.067, -0.052]* | 8.87 [7.78, 10.07]* | -38.28 [-40.77, -35.83]* |
| recurring | HH-R − DB-0.90 | -0.078 [-0.085, -0.070]* | 17.46 [15.79, 19.19]* | -40.82 [-43.13, -38.59]* |
| within_library_switch | HH-R − DB-0.95 | -0.034 [-0.046, -0.021]* | 5.04 [3.19, 6.96]* | -34.47 [-37.22, -31.77]* |
| outside_library_switch_m1 | HH-R − DB-0.95 | -0.031 [-0.036, -0.026]* | 4.64 [3.89, 5.41]* | -34.06 [-36.53, -31.68]* |
| outside_library_switch_m2 | HH-R − DB-0.95 | -0.027 [-0.035, -0.020]* | 4.08 [3.00, 5.27]* | -33.51 [-35.98, -31.09]* |
| recurring | HH-R − DB-0.95 | -0.024 [-0.031, -0.016]* | 5.36 [3.69, 7.08]* | -28.73 [-31.05, -26.49]* |
| within_library_switch | HH-R − DB-0.98 | 0.130 [0.118, 0.143]* | -19.52 [-21.40, -17.64]* | -9.90 [-12.60, -7.21]* |
| outside_library_switch_m1 | HH-R − DB-0.98 | -0.032 [-0.037, -0.026]* | 4.74 [3.90, 5.60]* | -34.16 [-36.66, -31.80]* |
| outside_library_switch_m2 | HH-R − DB-0.98 | 0.048 [0.040, 0.055]* | -7.20 [-8.31, -6.00]* | -22.23 [-24.66, -19.82]* |
| recurring | HH-R − DB-0.98 | 0.055 [0.047, 0.063]* | -12.42 [-14.11, -10.67]* | -10.95 [-13.23, -8.70]* |
| stationary | HH-R − WB-k20 |  |  | -38.80 [-41.18, -36.51]* |
| stationary | HH-R − DB-0.90 |  |  | -38.51 [-40.91, -36.20]* |
| stationary | HH-R − DB-0.95 |  |  | -38.52 [-40.93, -36.20]* |
| stationary | HH-R − DB-0.98 |  |  | -38.52 [-40.93, -36.20]* |

**Replay scope: HH-R full minus each alternative (and post_detection minus HH-A).** Positive = A higher. Mean paired difference over 200 seeds [95% bootstrap CI]. `*` = CI excludes 0.

| condition | A − B | return_post | regret_post | auc |
|---|---|---|---|---|
| within_library_switch | HH-R − HH-R|him_window | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| outside_library_switch_m1 | HH-R − HH-R|him_window | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| outside_library_switch_m2 | HH-R − HH-R|him_window | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| recurring | HH-R − HH-R|him_window | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| within_library_switch | HH-R − HH-R|post_detection | -0.095 [-0.107, -0.083]* | 14.20 [12.42, 16.06]* | -10.44 [-12.60, -8.22]* |
| outside_library_switch_m1 | HH-R − HH-R|post_detection | 0.093 [0.082, 0.105]* | -14.02 [-15.73, -12.35]* | 17.79 [15.95, 19.65]* |
| outside_library_switch_m2 | HH-R − HH-R|post_detection | 0.060 [0.049, 0.072]* | -9.04 [-10.76, -7.31]* | 12.80 [10.84, 14.82]* |
| recurring | HH-R − HH-R|post_detection | 0.006 [-0.003, 0.014] | -1.27 [-3.04, 0.58] | 2.92 [0.97, 4.84]* |
| within_library_switch | HH-R − HH-A | -0.095 [-0.107, -0.083]* | 14.18 [12.38, 16.08]* | -9.04 [-11.45, -6.56]* |
| outside_library_switch_m1 | HH-R − HH-A | 0.100 [0.089, 0.112]* | -15.02 [-16.76, -13.30]* | 20.18 [18.11, 22.32]* |
| outside_library_switch_m2 | HH-R − HH-A | 0.063 [0.051, 0.074]* | -9.39 [-11.12, -7.66]* | 14.54 [12.44, 16.63]* |
| recurring | HH-R − HH-A | 0.012 [0.004, 0.020]* | -2.77 [-4.60, -0.94]* | 5.54 [3.50, 7.57]* |
| within_library_switch | HH-R|post_detection − HH-A | 0.000 [-0.002, 0.003] | -0.02 [-0.39, 0.31] | 1.40 [0.58, 2.31]* |
| outside_library_switch_m1 | HH-R|post_detection − HH-A | 0.007 [0.002, 0.012]* | -1.00 [-1.74, -0.37]* | 2.39 [1.35, 3.55]* |
| outside_library_switch_m2 | HH-R|post_detection − HH-A | 0.002 [-0.001, 0.006] | -0.35 [-0.90, 0.19] | 1.73 [0.82, 2.74]* |
| recurring | HH-R|post_detection − HH-A | 0.007 [0.004, 0.010]* | -1.50 [-2.18, -0.83]* | 2.62 [1.75, 3.56]* |

**ε schedule: HH-R constant minus decay_reset.** Positive = A higher. Mean paired difference over 200 seeds [95% bootstrap CI]. `*` = CI excludes 0.

| condition | A − B | return_pre | return_post | auc |
|---|---|---|---|---|
| stationary | HH-R − HH-R|decay_reset | 0.020 [0.004, 0.037]* | — | 6.07 [1.27, 10.96]* |
| within_library_switch | HH-R − HH-R|decay_reset | 0.033 [0.016, 0.050]* | 0.035 [0.015, 0.057]* | 10.21 [6.75, 13.68]* |
| outside_library_switch_m1 | HH-R − HH-R|decay_reset | 0.033 [0.016, 0.050]* | -0.003 [-0.011, 0.007] | 4.54 [1.26, 7.91]* |
| outside_library_switch_m2 | HH-R − HH-R|decay_reset | 0.033 [0.016, 0.050]* | 0.011 [-0.002, 0.024] | 6.57 [3.36, 9.85]* |
| recurring | HH-R − HH-R|decay_reset | 0.027 [0.013, 0.041]* | 0.005 [-0.006, 0.016] | 3.18 [0.56, 5.82]* |
| stationary | HH-R − HH-R|him_window+decay_reset | 0.020 [0.004, 0.037]* | — | 6.07 [1.27, 10.96]* |
| within_library_switch | HH-R − HH-R|him_window+decay_reset | 0.033 [0.016, 0.050]* | 0.035 [0.015, 0.057]* | 10.21 [6.75, 13.68]* |
| outside_library_switch_m1 | HH-R − HH-R|him_window+decay_reset | 0.033 [0.016, 0.050]* | -0.003 [-0.011, 0.007] | 4.54 [1.26, 7.91]* |
| outside_library_switch_m2 | HH-R − HH-R|him_window+decay_reset | 0.033 [0.016, 0.050]* | 0.011 [-0.002, 0.024] | 6.57 [3.36, 9.85]* |
| recurring | HH-R − HH-R|him_window+decay_reset | 0.027 [0.013, 0.041]* | 0.005 [-0.006, 0.016] | 3.18 [0.56, 5.82]* |
