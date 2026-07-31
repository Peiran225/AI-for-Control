# Neural-PMP paper/released-code-informed track

This directory is a paper/released-code-informed isolated reimplementation of
**Pontryagin Optimal Control via Neural Networks**
([arXiv:2212.14566](https://arxiv.org/abs/2212.14566)). It does not execute the
historical external solver and must not be described as an author-code-derived
tumor result.

## Algorithmic contract

`core.py` implements the update equations used for the paper's discrete
Algorithm 1:

1. learn or supply a differentiable one-step dynamics map;
2. roll the state forward;
3. set the terminal costate to the terminal-cost gradient;
4. recurse the discrete Hamiltonian costates backward;
5. calculate the **raw** Hamiltonian action gradient;
6. update the action using the raw gradient and then project it onto the box.

The last step is exactly

```text
candidate = u - learning_rate * raw_dH_du
u_next = projection(candidate, action_lower, action_upper)
```

The gradient is never clipped to the action bounds.  This corrects the update
ordering defect in the released repository while following Eqs. (32)--(33) of
the paper.

## Upstream checkout provenance

The nested checkout audited for this track is
`external/NeuralPMP2024` at HEAD
`e8a6269976bfd77ce7f7000ca6ca9ea3b5022833`.  It is **tracked dirty** in the
project workspace (`Env.py`, `Solver.py`, and `main.py` are modified), so this
independent runner never imports or executes that worktree.

At the audited snapshot:

| file | HEAD blob SHA-256 | worktree SHA-256 |
|---|---|---|
| `NeuralPMP/Env/Env.py` | `b24990a3a4c6f4c647d4e47de6a782b9a5e79944b47e6b087a4f8a7504f4b40b` | `67e448e9552283ecce659b51eb4e0cd848f87bf5b0f2fc1aaf6df153cfeb33ea` |
| `NeuralPMP/Solver/Solver.py` | `de85626ef8bd42e6fcd03a87285e219a50aa9ff071d965b28d86d39966337da7` | `5bb1dcb619670d80b58df779cf3e467b47672e23bd2a05e00c504783cff39e24` |

The LQR sampling, matrices, bounds and two-layer architecture are audited from
the immutable HEAD `Env.py` blob.  The worktree `Env.py` primarily normalizes
newlines, imports `os`, and appends the tumor environment after the LQR block.

Both the released HEAD `Solver.py` and the current worktree clip the
Hamiltonian gradient using action bounds before the update.  The worktree then
adds action projection, but that does not remove the pre-update gradient-clipping
defect.  This runner instead implements the paper's Eqs. (32)--(33): use raw
`dH/du`, take the step, then project the action.

Every LQR run recomputes and records nested HEAD, tracked-dirty files, HEAD blob
OIDs/SHA-256s, worktree SHA-256s, diff SHA-256s/numstats, and the source policy
in both `run_config_and_summary.json` and `manifest.json`.  Thus later worktree
changes cannot silently inherit this audit statement.

## Evidence levels

- `run_lqr_smoke.py` exercises the complete learned-dynamics LQR pipeline at a
  small budget.  It uses the Appendix-C matrices/costs, state domain
  `[-5, 5]` and the two-hidden-layer ReLU dynamics architecture.  By default it
  follows the official code and samples dynamics-training actions in `[-5, 5]`;
  the controller feasibility bounds remain `[-100000, 100000]`.
- `run_lqr_full.py` uses the paper budget: 2,000 dynamics samples, 50,000
  dynamics epochs, 3,000 PMP updates at learning rate `1e-3`, and 10
  independent runs.  Every seed's dataset, dynamics checkpoint, control and
  history are retained; the output is a 10-run aggregate, not a best seed.
- `run_tumor.py --dynamics-mode learned` is an **algorithmic adaptation** to
  the tumor problem.  The paper did not contain this tumor
  environment, so this is not an original numerical experiment from the
  paper.
- `run_tumor.py --dynamics-mode exact` is an oracle-dynamics ablation using the
  same PMP controller optimization.

### Completed paper-budget LQR reimplementation

The full profile labelled `official-code` in the runner was completed on
2026-07-12 for seeds
`0,1,2,3,4,5,6,7,8,9`.  Each run used 2,000 dynamics samples, the fixed-final
50,000-epoch dynamics checkpoint, a zero control start, and the fixed-final
3,000th PMP iterate.  No validation metric or true LQR objective selected a
checkpoint, iterate, or seed.

The retained run is
`paper_runs/faithful_related_work/neural_pmp_lqr_full_10seed_server`.  Its true
LQR objective is `13.543236635770143 +/- 0.12694848037949266` (sample standard
deviation) across the ten declared seeds.  The independently computed finite-
horizon Riccati reference is `13.428949031211102`, so the mean relative gap is
`0.8510539752099477%`.  The server wall time was 14 minutes 22.99 seconds; the
companion execution record and raw timing log are retained under
`paper_runs/faithful_related_work/server_logs/`.

## Determinism and selection

All NumPy, Python, and Torch RNGs are seeded.  Offline dynamics train/validation
datasets are saved.  Exactly one dynamics model is trained per independent
seed, saved once, and shared by every control initialization for that seed.

The original-LQR runner uses the single declared zero start and the fixed
final iterate, so it performs no validation-based selection.  The tumor runner
does not use a fixed-final pipeline: (1) it selects the minimum-validation-MSE
dynamics checkpoint; (2) it selects a controller iterate within each start by
the native learned-dynamics validation objective; and (3) it selects one of
five starts using the same fixed eight-state controller-validation set as
layer 2. The best dynamics epochs for seeds 0/1/2 are `750/1000/650`. There is
no held-out native test set after the three selection layers. Neither runner ranks
seeds by performance.
`canonical_control.npz` always comes from `--canonical-seed` (default seed 0),
fixed before execution.  True/realized objectives are evaluated only after all
applicable within-seed decisions and the canonical seed are frozen.

The declared native protocol selects the `zero` start for all three tumor
seeds. Independent physical execution shows that these are the worst common-
`J` candidates within their five-start groups: selected
`J=561.187/483.122/539.609`, versus within-seed best
`404.319/405.749/407.247`. Across all 15 candidates, Spearman correlation
between the native score and common `J` is `-0.643`. Thus the exported result
is the output of the declared native-selection rule, not the best start under
the common evaluator.

The selected learned-model rollouts are not physical-state validations. At
least one component is negative at `164/97/168` of 201 time points for seeds
0/1/2, and their maximum absolute differences from physical re-execution are
`29.421/23.395/28.344`. The independent audit's `1.03e-4` state-discrepancy
bound applies only to saved physical breakpoint-state artifacts, not these
learned rollouts.

Both runners refuse to overwrite a non-empty output directory.

## Paper/code LQR sampling conflict

The official LQR environment samples both state and action training data as
`-5 + 10 * rand`, so the default `official-code` profile uses

```text
dynamics state samples:  [-5, 5]
dynamics action samples: [-5, 5]
controller bounds:       [-100000, 100000]
```

Appendix C text can be read as sampling actions over the controller bounds.
That alternative is available only as an explicitly labelled sensitivity:

```bash
python -m faithful_related_work.neural_pmp.run_lqr_full \
  --out-dir paper_runs/faithful_neural_pmp_lqr_paper_text_sensitivity \
  --sampling-profile paper-text-sensitivity
```

It uses dynamics action samples in `[-100000, 100000]` and must not be reported
as the primary released-code-informed profile.

The official pipeline also uses the dynamics model after the fixed 50,000-step
training budget and the control after the fixed 3,000 PMP updates. Validation
losses are retained only as diagnostics: neither dynamics checkpoints nor
controls are selected by validation performance in the LQR runners.

## Commands

Fast checks:

```bash
python -m faithful_related_work.neural_pmp.run_lqr_smoke \
  --out-dir /tmp/faithful_neural_pmp_lqr_smoke
python -m faithful_related_work.neural_pmp.run_lqr_full \
  --out-dir paper_runs/faithful_neural_pmp_lqr_full
python -m unittest tests.test_faithful_neural_pmp
```

Small learned-dynamics tumor smoke:

```bash
python -m faithful_related_work.neural_pmp.run_tumor \
  --out-dir /tmp/faithful_neural_pmp_smoke \
  --seeds 0 --starts zero,mid \
  --n 20 --m 5 \
  --dynamics-train-samples 128 \
  --dynamics-validation-samples 64 \
  --dynamics-epochs 20 \
  --dynamics-validation-interval 5 \
  --dynamics-patience 5 \
  --control-iters 10 --control-eval-interval 2
```

The LQR full command uses the paper-scale defaults.  A full tumor adaptation
should separately report its declared compute budget; it is not an original
paper experiment.

## Outputs

The LQR and tumor runners save:

- deterministic dynamics datasets and validation initial states;
- one dynamics checkpoint and training history per seed;
- one control checkpoint and optimization history per seed/start;
- `all_runs.csv`, with the shared dynamics-checkpoint hash;
- `canonical_control.npz` containing canonical `t` and interval `u`;
- the complete run configuration, selection declaration, and post-selection
  metrics;
- a SHA-256 artifact manifest.
