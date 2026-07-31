# Audited Adaptive HJB-NN reconstruction and tumor adaptation ([7])

This directory implements two deliberately separate claims for Nakamura-Zimmerer,
Gong, and Kang, *Adaptive Deep Learning for High-Dimensional
Hamilton--Jacobi--Bellman Equations*, SIAM J. Sci. Comput. 43(2), 2021,
[DOI 10.1137/19M1288802](https://doi.org/10.1137/19M1288802),
[arXiv:1907.05317](https://arxiv.org/abs/1907.05317).

“Reconstruction” is intentional: the original satellite RNG seed, final
adaptive data set, exact scaling provenance, and per-round schedule were not
released, so the from-scratch runs below cannot be claimed as exact numerical
replays.

## 1. Author satellite benchmark through a compatibility port

The checkout at `external/HJB_NN` is **not clean or byte-for-byte upstream**.
It is based on the recorded author revision but contains tracked changes. The
satellite commands use its audited TensorFlow-2 compatibility port of:

- satellite dynamics, costs, characteristic BVP data, and released split;
- joint value/costate supervision;
- `HJBnet_t0` architecture and full-batch L-BFGS-B objective;
- Algorithm 4.1 convergence test and author adaptive generator when multiple
  retraining rounds are requested.

Validate the released checkpoint on the independent `data_test.mat` BVP
trajectories:

```bash
python -m faithful_related_work.hjb_nn.runner validate-satellite \
  --out-dir paper_runs/faithful_hjb_nn_satellite_validation
```

Re-run the paper's noisy 10 Hz zero-order-hold closed-loop protocol and compare
the released script's native costs with an independent tight-tolerance
realization of the same feedback policies:

```bash
python -m faithful_related_work.hjb_nn.runner evaluate-satellite-closed-loop \
  --seed 0 \
  --out-dir paper_runs/faithful_hjb_nn_satellite_closed_loop_seed0
```

The paper does not identify Figure 3's random seed, so this is a deterministic
protocol reconstruction rather than an exact replay of its `12.67/12.52/15.95`
sample.  The manifest separately records four numerical quirks in the released
`simulate_noise.py`/`compute_cost` path and never substitutes its native number
for the independently integrated realized objective.

Retrain the released satellite benchmark from scratch:

```bash
python -m faithful_related_work.hjb_nn.runner retrain-satellite \
  --seed 0 --max-rounds 10 --min-rounds 1 --maxiter 100000 \
  --out-dir paper_runs/faithful_hjb_nn_satellite_retrain
```

The ceiling of ten rounds allows the paper's Algorithm 4.1 convergence rule to
stop the run; it does not force ten rounds.  The paper reports that its adaptive
satellite run stopped after seven rounds with 2,110 samples.  Its random seed,
final adaptive data set, exact initial-set/scaling provenance, and per-round
hyperparameter adjustments were not released, while the repository currently
defaults to only three rounds.  A
deterministic from-scratch run is therefore a **paper-configuration
reconstruction**, not a bitwise replay of Figure 2.  The released author-code/
paper-setting optimizer
budget is retained as `maxiter=maxfun=100000` per round.

`--warm-start-released` is available as a separately declared continuation
experiment. It is off by default and must not be presented as from-scratch
retraining.

To test whether merely forcing the paper's reported round count explains the
gap, run a separately labeled sensitivity with `--max-rounds 7 --min-rounds 7`.
Under the released doubling rule this ends at 4,096 samples, not the paper's
2,110, so it is diagnostic evidence rather than an exact replay.

## 2. Tumor adaptation

The tumor OCP is not an original-paper benchmark. The adaptation preserves the
paper's characteristic-BVP data, value+costate loss, full-batch L-BFGS, held-out
trajectory validation, and Algorithm 4.1 loop:

1. solve a small initial BVP set with time marching;
2. train the author `HJBnet` logic through the audited TensorFlow-2
   compatibility port with full-batch L-BFGS;
3. apply the author convergence/sample-size test (paper equations 4.8--4.9);
4. if it fails, independently sample candidate initial states from the full
   21-dimensional box;
5. select the largest predicted `||V_x(0,x_0)||_2`;
6. roll out the current NN feedback and use predicted value/costate as a BVP
   warm start;
7. append the resulting full characteristic trajectory and refine the same NN.

### A mathematically consistent smooth bounded control

For `p=u/U`, the adaptation declares

```text
L_tau(N,u) = beta.N + gamma*u
             + tau*U*[p log p + (1-p) log(1-p)].
```

Consequently the exact Hamiltonian minimizer is

```text
psi = gamma - sum_i phi_i lambda_i N_i
u_tau = U*sigmoid(-psi/tau).
```

The entropy term is included in `running_cost`, the BVP value-state ODE, and all
value labels. There is no singular-control blend. Each run reports the NN's
regularized native value separately from the common evaluator's unregularized
realized `J`.

Run the declared continuation/sweep (three seeds by default):

```bash
python -m faithful_related_work.hjb_nn.runner run-tumor \
  --taus 10,5,2 --seeds 0,1,2 \
  --train-trajectories 16 --validation-trajectories 16 \
  --max-rounds 3 --min-rounds 2 --maxiter 5000 \
  --bvp-max-nodes 100000 --adaptive-bvp-max-nodes 20000 \
  --time-march-steps 16 \
  --out-dir paper_runs/faithful_hjb_nn_tumor
```

The BVP solver starts at the largest declared `tau` and uses geometric
temperature continuation to the target value. The declared tumor-adaptation
optimizer budget is `maxiter=5000` and `maxfun=15000` per round; this is not the
original satellite paper's `100000/100000` budget. Every tau/seed run is
retained; unregularized realized `J` is never available to training, stopping,
checkpoint, or run selection.

The full profile retains `tol=1e-3` but uses 16 horizon-marching stages and a
100,000-node ceiling.  An initial 8-stage/20,000-node server attempt was retained
as a failed numerical-reliability pilot: multiple declared seeds exhausted the
mesh ceiling before training.  The larger uniform solver budget is applied to
every tau/seed run; it is not a seed-specific rescue or result-selection rule.
NN-warm-start adaptive candidates use a separate 20,000-node guard: candidates
that would refine beyond it are recorded as failed attempts and the author-style
candidate loop continues. Successful adaptive characteristics in the server
preflight used only 102--104 nodes; allowing pathological candidates to expand
to 100,000 nodes instead caused sparse-LU allocation failures without producing
accepted data.

For long server runs, execute isolated workers so one failed BVP cannot erase
the other declared runs.  Each worker uses the same defaults as the sweep and
retains a machine-readable failure manifest on error:

```bash
python -m faithful_related_work.hjb_nn.runner run-tumor-one \
  --tau 10 --seed 0 \
  --out-dir paper_runs/faithful_hjb_nn_tumor/tau_10/seed_0
```

After every declared `tau`/seed worker succeeds, verify their artifact hashes,
hash each worker manifest itself, and create the sweep root:

```bash
python -m faithful_related_work.hjb_nn.runner aggregate-tumor \
  --taus 10,5,2 --seeds 0,1,2 \
  --out-dir paper_runs/faithful_hjb_nn_tumor
```

Before using the aggregate in a report, run the independent final-protocol
audit. It re-transcribes the dynamics and both objectives without importing
the HJB evaluator, re-integrates every ZOH interval, reconstructs every seeded
train/validation split, verifies BVP residual histories, and checks all frozen
source and artifact hashes:

```bash
python scripts/audit_hjb_nn_final_results.py \
  --aggregate-manifest paper_runs/faithful_related_work/hjb_nn_tumor_full_final_v3/manifest.json \
  --out-dir paper_runs/faithful_related_work/hjb_nn_tumor_full_final_v3_independent_audit
```

`--adaptive-max-failures` changes only the declared retry budget for the
author-style NN warm-start candidate loop.  It never falls back to a different
BVP initializer after a failed candidate.

During collocation Newton iterations only, the dynamics have a documented
positive-orthant domain extension at `1e-12` because intermediate Newton
iterates can be nonphysical. It is inactive on accepted data: the generator
rejects any converged characteristic containing `N_i <= 0`.

An executable small-budget wiring check is available:

```bash
python -m faithful_related_work.hjb_nn.runner run-tumor \
  --taus 10,5 --seeds 0 --smoke \
  --out-dir /tmp/faithful_hjb_nn_smoke
```

The smoke profile is not a scientific result.

## TensorFlow-2 compatibility provenance

The installed TensorFlow 2 runtime no longer provides
`tensorflow.contrib.opt.ExternalOptimizerInterface`, so this checkout cannot
execute the historical source verbatim. Its tracked port:

- imports `tensorflow.compat.v1` and disables v2 behavior;
- replaces the removed `tensorflow.contrib` optimizer wrapper with a local
  `ScipyOptimizerInterface` that retains the same packed-variable SciPy
  L-BFGS-B objective/gradient mechanism;
- allows the tumor control graph to receive both value gradient and state,
  while retaining the one-argument fallback used by the satellite problem;
- also contains tracked `generate.py`, `train.py`, problem-template, and
  problem-selection changes plus an untracked tumor example directory.

Consequently the valid claim is “author HJBnet architecture family,
released data/checkpoint, or training logic through an audited compatibility
port,” not “unchanged official code” or “exact numerical replay.” Every manifest records `upstream_tracked_dirty: true`, the complete Git
status, per-file `git diff --name-status` and `--numstat`, a hash of the unified
tracked diff, all untracked paths, and hashes of imported files.

## Artifacts and provenance

Each tumor run writes:

- `dataset_train_initial.npz`, `dataset_train_final.npz`, and the trajectory-
  disjoint `dataset_validation.npz`;
- one `.mat` checkpoint after every L-BFGS round plus final `V_model.mat`;
- candidate states, predicted gradients, selected indices, and NN-warm-start
  BVP metadata for every adaptive event;
- `history.json`/`history.csv` with train/validation errors and convergence
  decisions;
- `canonical_control.npz` with canonical `t,u`, learned-feedback rollout,
  regularized native value prediction, independently integrated regularized
  realized `J` with terminal/running/entropy decomposition, native prediction
  error, and the common unregularized realized `J`;
- `manifest.json` with seeds, budgets, command, revisions, source hashes,
  checkpoint hashes, and the frozen no-realized-J selection rule.

The runner does not mutate the checkout while an experiment is executing. The
executed checkout already contains the recorded compatibility patch and dirty
tree described above. Each manifest therefore records its base Git revision,
the dirty-tree state, and frozen source snapshots; the base revision alone must
never be interpreted as the exact executed source.
