| method | big (8 pairs x 64 rollouts) | mid (2 pairs x 8 rollouts) | small (1 pair x 1 rollout) |
|---|---|---|---|
| oracle_lqr_zoh_k | 0.76 [0.76, 0.76] | 0.76 [0.76, 0.76] | 0.76 [0.76, 0.76] |
| zero_control | 3.08e+04 [3.08e+04, 3.08e+04] | 3.08e+04 [3.08e+04, 3.08e+04] | 3.08e+04 [3.08e+04, 3.08e+04] |
| decoupled_pca | 3.18e+04 [3.04e+04, 4.65e+04] | 3.18e+04 [3.04e+04, 4.65e+04] | 3.18e+04 [3.04e+04, 4.65e+04] |
| decoupled_rrr | 2.68e+04 [2.29e+04, 4.99e+04] | 2.68e+04 [2.29e+04, 4.99e+04] | 2.68e+04 [2.29e+04, 4.99e+04] |
| mlo_spsa_P_only | 2.27e+04 [2.2e+04, 2.71e+04] | 2.2e+04 [2.18e+04, 2.46e+04] | 2.61e+04 [2.47e+04, 2.81e+04] |
| mlo_spsa_E_only | 2.95 [1.31, 8.24e+03] | 23.2 [17.1, 40.5] | 1.45e+04 [8.08e+03, 3.93e+04] |
| mlo_spsa | 1.49 [1.41, 2.18] | 4.34e+04 [2.03, 1.31e+05] | 3.08e+04 [6.33e+03, 3.37e+04] |
| model_free_spsa | 1.99 [1.92, 2.72] | 5.17 [4.78, 17.7] | 41.1 [10.4, 71.9] |
| mlo_oracle | 1.26 [1.24, 1.26] | 1.3 [1.27, 1.31] | 1.45 [1.42, 1.66] |

- `big`: 3 seeds; real env steps per outer step = 122880; exploration data = 6000 steps; outer steps = 200
  - relevance of E row-space (1 = fully inside cost-relevant subspace): RRR init 0.22 -> MLO-SPSA 0.32; alignment of P E with unstable modes: RRR 0.02 -> MLO-SPSA 0.49
- `mid`: 3 seeds; real env steps per outer step = 3840; exploration data = 6000 steps; outer steps = 200
  - relevance of E row-space (1 = fully inside cost-relevant subspace): RRR init 0.22 -> MLO-SPSA 0.38; alignment of P E with unstable modes: RRR 0.02 -> MLO-SPSA 0.29
- `small`: 3 seeds; real env steps per outer step = 240; exploration data = 6000 steps; outer steps = 200
  - relevance of E row-space (1 = fully inside cost-relevant subspace): RRR init 0.22 -> MLO-SPSA 0.30; alignment of P E with unstable modes: RRR 0.02 -> MLO-SPSA 0.09
