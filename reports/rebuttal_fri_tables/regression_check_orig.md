Seeds 0..199 (200); tolerance 1e-6.

| quantity | n compared | bitwise | max abs diff | verdict |
|---|---|---|---|---|
| auc vs cumulative_return | 5000 | yes | 0 | PASS |
| regret_post vs post_switch_regret | 4000 | yes | 0 | PASS |
| recovery_time vs adaptation_delay | 4000 | yes | 0 | PASS |
| n_detections/T vs false_alarm_rate | 2000 | no | 8.33e-17 | PASS |
| n_accept_reuse | 2000 | yes | 0 | PASS |
| n_accept_merge vs revise_accepted | 2000 | yes | 0 | PASS |
| n_accept_construct | 2000 | yes | 0 | PASS |
| n_revisions | 2000 | yes | 0 | PASS |
| n_policy_updates | 2000 | yes | 0 | PASS |
| return_pre vs return_before | 5000 | yes | 0 | PASS |
| return_post vs return_after | 5000 | yes | 0 | PASS |

OVERALL: PASS
