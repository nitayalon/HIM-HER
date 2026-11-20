Within-library, HH-R `replay_scope=full`, seeds 0–19, `log_level=full`: 84 acceptances in total, 23 before the switch (t < 150) and 61 at or after it.

| subset | n acceptances | mean n_replayed | mean fraction of replayed transitions from before the switch | mean log-lik/step of those pre-switch transitions: accepted model | : true pre-switch model M_L | : incumbent before acceptance |
|---|---|---|---|---|---|---|
| all acceptances | 84 | 17.01 | 0.637 | -0.626 | -0.455 | -0.575 |
| acceptances at t ≥ 150 | 61 | 20.00 | 0.500 | -0.670 | -0.312 | -0.459 |
| acceptances at t ≥ 150 whose replay includes pre-switch data | 58 | 20.00 | 0.526 | -0.670 | -0.312 | -0.459 |

First acceptance after the switch, per seed (20 of 20 seeds have one): mean t_rel_switch 4.9 (median 5); mean fraction of replay from before the switch 0.705; accepted operator counts {'merge': 13, 'construct': 4, 'reuse': 3}; accepted model = M_R (0.9, 0.9) in 0 of them. Over all post-switch acceptances, 4 of 61 accept M_R.
