# Three-Case Experiment Guide

This file is the navigation entry point for the current `n=800` comparison.
The three cases use the same physical tumor problem, control bounds, and
evaluation conventions.

## 1. Case definitions

| Case ID | Report label | Training entry point |
|---|---|---|
| `time_only` | Transformer `u(t)` | `train_paper_pmp_kkt.py` followed by the resolution curriculum |
| `feedback_cf` | Case 1 `u(N,t)` | `scripts/train_feedback_section5.py --option cf` |
| `feedback_der` | Case 2 `u(N,t)` | `scripts/train_feedback_section5.py --option der` |

Case 1 fits the admissible closed-form singular-control candidate.  Case 2
uses residuals involving the switching function `H_u`, its first and second
time derivatives, and the Legendre--Clebsch term.  Both feedback cases can
also include the complete reduced-gradient residual with the state dependence
`N=N(u)` retained.

## 2. Shared model and numerical implementation

- `configs/tumor_nominal.json`: common problem parameters and evaluation
  tolerances.
- `tumor_problem.py`: tumor dynamics, costate diagnostics, and segmented
  high-accuracy zero-order-hold evaluation.
- `train_paper_pmp_kkt.py`: common parameter construction, time features,
  Transformer architecture, and PMP/KKT quantities.
- `scripts/feedback_section5_rk4_reference.py`: RK4 state recursion and its
  matching discrete adjoint for the feedback cases.
- `scripts/diagnose_reduced_objective_hessian.py`: reduced objective,
  projected first-order residual, and Hessian utilities.

## 3. Training entry points

### 3.1 Time-only `u(t)`

The report-compatible training chain starts with the PMP/KKT Transformer on
the `n=200` grid:

```bash
.venv/bin/python train_paper_pmp_kkt.py \
  --model transformer \
  --n 200 \
  --epochs 1200 \
  --lr 5e-4 \
  --singular_eps 0.1 \
  --singular_tau 0.03 \
  --smooth_weight 3 \
  --seed 4 \
  --device cpu \
  --float64 \
  --out_dir runs/time_base_seed4
```

It is then continued to `n=800` using the projected reduced-gradient residual
with annealed PMP and smoothness terms:

```bash
.venv/bin/python scripts/train_teacher_free_resolution_curriculum.py \
  --start-checkpoint runs/time_base_seed4/best_pmp_kkt.pt \
  --out-dir runs/time_curriculum \
  --seed 106 \
  --scale 1.08 \
  --temperature 0.7 \
  --epochs 300,220,180 \
  --learning-rates 8e-5,3e-5,1e-5 \
  --linf-weights 0.2,0.5,1 \
  --pmp-weights 0.02,0.004,0 \
  --smooth-weights 0.0001,0.00002,0
```

This produces the compatible intermediate checkpoint
`runs/time_curriculum/stage_3_n800/selected_checkpoint.pt`; it is not the
selected checkpoint used for the reported figures.  The reported time-only
checkpoint also passes through projected-fixed-point, strict-KKT, and L-BFGS
continuation stages implemented in:

```text
scripts/train_teacher_free_resolution_curriculum.py
scripts/continue_teacher_free_projected_fixed_point.py
scripts/continue_teacher_free_strict_kkt.py
scripts/continue_teacher_free_strict_optimality.py
```

Each continuation script accepts the preceding checkpoint explicitly.  Run
the corresponding script with `--help` for its stage-specific settings.
`scripts/train_teacher_free_native_n800.py` is a separate random-start
experiment; its checkpoint format is not the input to the three-case figure
generators.

### 3.2 Case 1 `u(N,t)`

The following commands show the required routing and provide a short training
run.  They are not the complete hyperparameter specification of the selected
reported checkpoints.

```bash
.venv/bin/python scripts/train_feedback_section5.py \
  --option cf \
  --loss_variant literal \
  --n 800 \
  --time_checkpoint /path/to/time_only_checkpoint.pt \
  --out_dir runs/feedback_case1 \
  --training_integrator rk4 \
  --state_feature_mode relative_nominal \
  --freeze_time_branch \
  --full_gradient_weight 50 \
  --full_gradient_residual projected \
  --float64
```

### 3.3 Case 2 `u(N,t)`

```bash
.venv/bin/python scripts/train_feedback_section5.py \
  --option der \
  --loss_variant literal \
  --n 800 \
  --time_checkpoint /path/to/time_only_checkpoint.pt \
  --out_dir runs/feedback_case2 \
  --training_integrator rk4 \
  --state_feature_mode burden_composition \
  --freeze_time_branch \
  --full_gradient_weight 50 \
  --full_gradient_residual projected \
  --float64
```

The selected experiments use additional loss weights and continuation
settings.  Case 2 also starts from an earlier state-branch checkpoint.  These
are command-line options in `train_feedback_section5.py`; the script records
the complete argument set in each checkpoint and output summary.

## 4. Unified `H`, `H_u`, and `H_uu` diagnostics

The matched three-case generator realizes one physical interval-control vector
for each policy and evaluates:

```text
H(t)                         instantaneous Hamiltonian along the trajectory
dF_h/du                      full reduced gradient, including N=N(u)
d^2 F_h/du^2                 full reduced Hessian
```

Run it with the three selected checkpoints:

```bash
.venv/bin/python scripts/generate_teacher_facing_three_case_n800.py \
  --time_checkpoint /path/to/time_only_checkpoint.pt \
  --cf_checkpoint /path/to/case1_checkpoint.pt \
  --der_checkpoint /path/to/case2_checkpoint.pt \
  --out_dir runs/three_case_diagnostics
```

The continuous switching-function diagnostics are implemented separately in
`scripts/generate_three_case_hamiltonian_results.py`.  The full control-vector
gradient and Hessian calculations are in
`scripts/generate_three_case_full_derivative_results.py`.

## 5. Nominal versus resistant-heavy comparison

This command generates the same set of state and control diagnostics for the
nominal and resistant-heavy initial states, with the two trajectories
overlaid consistently for each case:

```bash
.venv/bin/python scripts/generate_two_state_three_case_main_figures.py \
  --time-checkpoint /path/to/time_only_checkpoint.pt \
  --cf-checkpoint /path/to/case1_checkpoint.pt \
  --der-checkpoint /path/to/case2_checkpoint.pt \
  --out-dir runs/two_state_three_case
```

Use a new or empty output directory for either generator; result files are not
silently overwritten.

## 6. Efficiency benchmarks

- `scripts/benchmark_time_only_method_efficiency.py` compares construction,
  loaded inference, and matched rollout timing for the time-only methods.
- `scripts/benchmark_feedback_vs_state_direct.py` compares a loaded feedback
  policy with nominal-schedule replay and state-specific direct recomputation.
- `scripts/benchmark_time_only_inference.py` provides a focused loaded-network
  inference benchmark.

Every benchmark writes its timing protocol and run metadata with the numeric
results.  Checkpoint and reference-solution paths can be supplied explicitly;
for example:

```bash
.venv/bin/python scripts/benchmark_time_only_inference.py \
  --checkpoint /path/to/time_only_checkpoint.pt \
  --out runs/benchmarks/time_only_inference.json

.venv/bin/python scripts/benchmark_feedback_vs_state_direct.py \
  --feedback /path/to/case2_checkpoint.pt \
  --nominal-direct /path/to/direct_n800_solution.npz \
  --out-dir runs/benchmarks/feedback_vs_direct
```

These commands cannot run from a fresh clone alone because the binary
artifacts are not versioned.  The focused inference benchmark requires a
time-only checkpoint.  The feedback-versus-direct benchmark requires a Case 2
checkpoint and the nominal direct `.npz` solution.  The complete time-only
benchmark additionally requires the refined checkpoint, native-`n=800`
Direct-J checkpoint, PMP checkpoint, and training-timing manifest; its MLP and
Neural-PMP inputs are optional.  Run each script with `--help` for the complete
artifact interface.

## 7. Tests

```bash
.venv/bin/python -m pytest -q
```

The tests cover the common tumor evaluator, feedback losses, RK4 discrete
adjoint, checkpoint evaluation, and the shared bounded-control utilities.

## 8. Earlier experiments

The following files are retained as earlier baselines and are not the primary
entry points for the current three-case results:

```text
scripts/train_openloop.py
scripts/train_feedback.py
scripts/refine_feedback.py
scripts/evaluate.py
scripts/run_example.sh
```
