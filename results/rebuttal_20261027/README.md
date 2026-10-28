# Additional numerical evaluations (2026-10-27)

These files record fixed-checkpoint evaluations and one recorded-budget
Direct-initialized training run. They do not replace the submitted checkpoints
or change the paper's tables.

## Evaluation conventions

- Physical weights: `alpha=1`, `beta=40`, `gamma=8000`; horizon `T=10`;
  21 phenotype components and controls in `[0,3]`.
- Nominal initial state: all components are 10.
- Structured shift: `N_i(0)=8+0.2*(i-1)`, for `i=1,...,21`.
- Task `J`: the original `n=800` evaluator with four classical-RK4 substeps
  per control interval and its original left-endpoint control execution.
- Scalar PMP diagnostics: the original S4 evaluator, with 25,600 fixed RK4
  steps, the open-loop adjoint along each realized trajectory, and 16,120
  points in `[1.5,8)` after excluding the `n=800` support nodes from `T32`.
- The original execution rule directly queries Transformer raw logits on
  `T32` against a fixed 801-token support context, and uses their PCHIP
  extension at internal ODE stages before the bounded action transform.
  Feedback corrections use each current stage state.
- Additional support-key/value caching changes only repeated computation;
  `query_cache_checks.json` records comparisons with the original query.
- `epsilon = sqrt(RMS(psi)^2 + RMS(dot_psi)^2 + RMS(ddot_psi)^2)`.
  Each state has its own time RMS; state and time axes are not pooled.

This is the 16,120-point S4 convention, not the 12,480-point
`T32 minus T8` convention used in other supplementary studies. Dense-rollout
objectives in raw diagnostic files are auxiliary and are not the task `J`.

## Files

- `fixed_policy_metrics.csv`: frozen nominal and structured-shift diagnostics.
- `original_seed_metrics.csv`: the original three training replicates,
  evaluated without retraining or replacing a checkpoint.
- `heldout_per_state.csv`: all 128 original initial-state IDs for each policy,
  original task objectives, scalar residuals and state-specific Direct gaps.
- `heldout_initial_states.csv`: the original 128 synthetic initial states.
- `objective_components.csv`: weighted terminal, running-population and dosing
  contributions, with unrounded reconstruction errors.
- `noncf_before_after.csv`: supervised-only and scalar-refined snapshots from
  a single chain without CF-trained initialization weights.
- `original_stage_timings.csv`: newly timed original feedback stages, labeled
  separately from historical logs and from total tuning cost.

Sample standard deviations over the three original training replicates and
statistics across 128 different initial states are different quantities.
The per-state tables retain all states. The original implementation did not
establish a per-state singular-interval detector: its figure markers select
the largest adjacent control change in early/late windows. These markers do
not establish a singular arc. No missing boundary tolerance is invented.
Time's original grid diagnostics use exact endpoint equality, and the same
zero-tolerance check is reported separately in `time_exact_bound_check.json`.
No corresponding original CF/DER boundary selector was verified.

The single non-CF run removes CF-trained ancestry, retains Direct supervision,
and uses the recorded stage seeds and update budgets. Its fixed training
states are nominal and the structured radius-0.10 state; radius 0.20 is used
only for evaluation. The final scalar stages use a fixed core rather than a
dynamic DER gate. No gate is added solely to create a sensitivity result.
The legacy `euler_pmp_kkt_opt_gap` score is logged only after temporal
checkpoint selection; it does not enter training, selection, or the reported
before/after metrics.

Timing measurements use the original saved stage inputs and configurations;
their resulting weights are not substituted for submitted models. They do
not reconstruct missing historical hyperparameter-search cost.

Large checkpoints and dense trajectory arrays are not versioned, consistently
with this branch's existing artifact policy. The scripts take the retained
artifact paths; lightweight numerical records and checkpoint hashes are
provided here. Internal review documents are not included.
