# Supplementary controlled experiments

This directory contains the lightweight numerical summaries added to the
AAAI-27 supplementary material. Large checkpoints, trajectory arrays, and
generated output directories are intentionally excluded from version control.
Every reproduction command below therefore takes explicit artifact paths.

## Strict diagnostic convention

The scalar diagnostics use the physical objective weights and the strict
off-grid set

`[1.5, 8.0) intersect (T_32 minus T_8)`.

This gives 12,480 evaluation times that are absent from the `T_8` refinement
grid. The reported combined residual is

`epsilon = sqrt(RMS(psi)^2 + RMS(dot_psi)^2 + RMS(ddot_psi)^2)`.

## Near-equal-objective structure study

[`near_equal_objective.csv`](near_equal_objective.csv) records the locked
PMP-Time control and four signed smooth perturbations. The physical objective
changes by less than `5e-4`, while the higher-order residuals change by orders
of magnitude.

```bash
python scripts/evaluate_near_equal_objective_structure.py \
  --baseline-npz /path/to/timeseries.npz \
  --baseline-summary /path/to/summary.json \
  --out-dir output/near_equal_objective
```

## Matched singular-loss ablation

[`nonuniform_beta_loss_ablation.csv`](nonuniform_beta_loss_ablation.csv)
compares first-order, state-only closed-form, and higher-order derivative
losses with the same direct-initialized Transformer, 65-parameter output head,
LM settings, and update budget. The physical weights are
`alpha_i=1`, `beta_i=30,...,50`, and `gamma=8000`. Checkpoint selection uses
the objective optimized by the corresponding arm.

Example 12-update commands are:

```bash
# First-order arm
python scripts/run_nonuniform_beta_loss_ablation.py \
  --start-checkpoint /path/to/start.pt \
  --out-dir output/nonuniform_beta/first_order_12 \
  --iterations 12 --w0 1 --w1 0 --w2 0 \
  --selection-metric training-objective

# State-only closed-form arm
python scripts/run_nonuniform_beta_loss_ablation.py \
  --start-checkpoint /path/to/start.pt \
  --out-dir output/nonuniform_beta/closed_form_12 \
  --iterations 12 --singular-loss cf-state \
  --selection-metric training-objective

# Higher-order derivative arm
python scripts/run_nonuniform_beta_loss_ablation.py \
  --start-checkpoint /path/to/start.pt \
  --out-dir output/nonuniform_beta/higher_order_12 \
  --iterations 12 --w0 1 --w1 1 --w2 1 \
  --selection-metric training-objective
```

Evaluate each retained checkpoint with the independent fixed-step evaluator:

```bash
python scripts/evaluate_nonuniform_beta_loss_ablation.py \
  --candidate-name higher_order_12 \
  --checkpoint output/nonuniform_beta/higher_order_12/selected_checkpoint.pt \
  --direct-npz /path/to/direct_n800_nonuniform_beta.npz \
  --source-file output/nonuniform_beta/higher_order_12/summary.json \
  --output output/nonuniform_beta/higher_order_12_audit.json
```

## Per-state computation time

[`fresh_direct_timing_summary.csv`](fresh_direct_timing_summary.csv) compares a
fresh warm-started Direct solve with loaded PMP-CF and PMP-DER closed-loop
rollouts on the same 128 held-out initial states. The state seed is `20260720`,
the perturbation radius is `0.20`, and the execution grid has 800 intervals.
[`fresh_direct_per_sample_timing.csv`](fresh_direct_per_sample_timing.csv)
contains the per-state Direct timings and projected-gradient checks.

The experiment is reproduced in three steps:

```bash
python scripts/table1_direct_heldout_runner.py \
  --direct /path/to/nominal_direct.npz \
  --out-dir output/direct_heldout

python scripts/benchmark_feedback_heldout_timing.py \
  --cf /path/to/pmp_cf.pt --der /path/to/pmp_der.pt \
  --direct-state-dir output/direct_heldout/samples \
  --out-dir output/feedback_timing

python scripts/aggregate_fresh_direct_timing.py \
  --source output/direct_heldout \
  --feedback-summary output/feedback_timing/summary.json \
  --out-dir output/timing_summary
```

Direct timing contains the optimizer but excludes the later dense diagnostic.
Learned-policy timing excludes checkpoint loading and includes 800 online
state-conditioned queries plus the corresponding closed-loop rollout.
