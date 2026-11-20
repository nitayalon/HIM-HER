Seeds 0..9 (10); tolerance 1e-6.

| quantity | n compared | bitwise | max abs diff | verdict |
|---|---|---|---|---|
| auc vs cumulative_return | 250 | yes | 0 | PASS |
| regret_post vs post_switch_regret | 200 | yes | 0 | PASS |
| recovery_time vs adaptation_delay | 200 | yes | 0 | PASS |
| n_detections/T vs false_alarm_rate | 100 | no | 6.94e-17 | PASS |
| n_accept_reuse | 100 | yes | 0 | PASS |
| n_accept_merge vs revise_accepted | 100 | yes | 0 | PASS |
| n_accept_construct | 100 | yes | 0 | PASS |
| n_revisions | 100 | yes | 0 | PASS |
| n_policy_updates | 100 | yes | 0 | PASS |
| return_pre vs return_before | 250 | yes | 0 | PASS |
| return_post vs return_after | 250 | yes | 0 | PASS |
| per-step theta0 vs model_trace_raw | 45000 | no | 9.71e-17 | PASS |
| per-step theta1 vs model_trace_raw | 45000 | no | 9.71e-17 | PASS |
| per-step kld vs kld_raw | 45000 | no | 8.88e-16 | PASS |

OVERALL: PASS
