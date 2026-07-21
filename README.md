# Transformer Optimal Control: Three-Case Experiments

This branch contains the code used to train and evaluate three control cases
on one tumor model and a common numerical grid:

| Case | Policy | Optimality-condition formulation |
|---|---|---|
| Time-only | `u(t)` | projected full reduced-gradient and PMP/KKT residuals |
| Case 1 | `u(N,t)`, option `cf` | closed-form singular-control candidate |
| Case 2 | `u(N,t)`, option `der` | residuals based on `H_u`, its time derivatives, and the Legendre--Clebsch term |

Case 1 and Case 2 share the same feedback architecture.  They differ in the
optimality-condition loss used for training.

## Start here

- [`THREE_CASE_GUIDE.md`](THREE_CASE_GUIDE.md) gives the code map, execution
  order, and commands for the current experiments.
- [`tumor_problem.py`](tumor_problem.py) defines the shared tumor dynamics and
  high-accuracy zero-order-hold evaluator.
- [`train_paper_pmp_kkt.py`](train_paper_pmp_kkt.py) contains the shared
  time-only Transformer and PMP/KKT implementation.
- [`scripts/train_feedback_section5.py`](scripts/train_feedback_section5.py)
  is the common training entry point for Case 1 and Case 2.
- [`scripts/generate_teacher_facing_three_case_n800.py`](scripts/generate_teacher_facing_three_case_n800.py)
  computes the matched three-case Hamiltonian, full reduced gradient, and
  reduced Hessian diagnostics.
- [`scripts/generate_two_state_three_case_main_figures.py`](scripts/generate_two_state_three_case_main_figures.py)
  generates the nominal versus resistant-heavy comparison for all three cases.

## Environment and checks

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Generated experiment directories, selected `.pt` checkpoints, and large
`.npz` arrays are not versioned on this branch.  Scripts that reproduce the
selected figures therefore take explicit checkpoint paths.  The source code,
configuration, and lightweight tests are versioned.

Earlier open-loop and feedback experiments remain in the repository for
reference.  The files listed above and in `THREE_CASE_GUIDE.md` are the entry
points for the current three-case comparison.
