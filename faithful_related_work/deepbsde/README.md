# DeepBSDE official-benchmark and tumor-adaptation track

This directory integrates Han, Jentzen and E's unchanged DeepBSDE solver for an
official HJB-LQ benchmark and a separate local tumor equation. It imports
`external/DeepBSDE/solver.py`, so
the following original mechanisms are retained:

- a trainable scalar `Y_0` and trainable vector `Z_0`;
- a separate feed-forward `Z` network at every later time step;
- forward Euler propagation of the BSDE;
- terminal-condition matching loss (including the official robust tail);
- the official Adam optimizer and learning-rate schedule.

The original-paper HJB-LQ track uses both the official `HJBLQ` equation and
the official solver.  It has two deliberately separate modes:

- `hjb-lq-smoke` is a tiny wiring test and **is not a numerical reproduction**;
- `hjb-lq-full` is the declared five-seed 100-dimensional numerical
  reproduction and reads the official checkout's
  `external/DeepBSDE/configs/hjb_lq_d100.json` unchanged.

The tumor problem is an equation adaptation only; it does not change the
solver.

## Upstream checkout provenance

`external/DeepBSDE` is a nested Git checkout at commit
`e76fed80b1995daf4a0dd6d3e2cf64a0931dabda`.  Its current worktree is
intentionally recorded as `upstream_tracked_dirty: true`:

- tracked `equation.py` has 62 appended lines defining a legacy `TumorHJB`;
- the bytes from the beginning of `equation.py` through the original upstream
  EOF are identical to the `HEAD:equation.py` blob;
- the original `HJBLQ` class is therefore unchanged;
- `solver.py` and `configs/hjb_lq_d100.json` are byte-identical to their
  upstream `HEAD` blobs;
- `configs/tumor_hjb_d21.json` is untracked in that nested checkout.

The manifests do **not** claim that the entire worktree `equation.py` is
unchanged.  They record the nested HEAD, porcelain status, numstat summary,
Git blob IDs, upstream/worktree SHA-256 values, original-prefix length/hash,
and appended-region hash.  Before any HJB-LQ full run, the runner fails closed
unless the only tracked difference is the recognized EOF-only `TumorHJB`
append and the original equation prefix, solver, and official HJB-LQ config
still match `HEAD` exactly.

The tumor-adaptation runner never imports the appended legacy class.  It uses
the independently derived log-state equation in this directory's
`equations.py`, while retaining the verified upstream solver.

The retained tumor worker manifests bind the official external solver and
equation sources, but they predate binding the local adaptation files
`faithful_related_work/deepbsde/runner.py` and `equations.py`. The independent
audit records current hashes for those files, which detects future drift but
does not prove their execution-time identity for the retained workers. Current
runner code includes these local source records for future reruns.

## Original-paper HJB-LQ benchmark

For the official 100-dimensional problem, equations (13)--(14) of the paper
give

```text
u_t + Delta(u) - lambda ||grad(u)||^2 = 0,
g(x) = log((1 + ||x||^2)/2),
u(0,0) = -(1/lambda) log E[exp(-lambda g(sqrt(2) W_T))].
```

At `dim=100`, `T=1`, and `lambda=1`, deterministic generalized
Gauss--Laguerre quadrature gives

```text
u(0,0) = 4.590161724604863...
```

The full runner records absolute and relative error at every logged training
step, retains all seeds `[0,1,2,3,4]`, and compares the aggregate only after
all five runs complete.  The paper reports a five-run relative error of 0.17%;
that number is contextual metadata, never a tuning target.  See the
[original paper](https://arxiv.org/abs/1707.02568), especially equations
(13)--(14) and Figure 2.

## Tumor HJB derivation

Set `x_i = log(N_i)`.  The deterministic controlled dynamics are

```text
dx_i/dt = r_i - M_i G(exp(x)) - phi_i u,
G(exp(x)) = log(1 + mean_i exp(x_i)).
```

For each positive viscosity `sigma`, use the reference SDE

```text
dX_i = [r_i - M_i G(exp(X))] dt + sigma dW_i.
```

The corresponding semilinear parabolic HJB is

```text
V_t + b0(x)·grad(V) + sigma^2/2 Delta(V)
    + beta·exp(x)
    + min_{0 <= u <= U} u [gamma - phi·grad(V)] = 0,
V(T,x) = alpha·exp(x).
```

Under the DeepBSDE convention, `Z = sigma grad(V)`.  Thus

```text
psi = gamma - phi·Z/sigma,
u*  = 0  if psi >= 0,
u*  = U  if psi < 0.
```

This endpoint rule is the exact box-constrained Hamiltonian minimizer.  There
is no sigmoid smoothing, entropy surrogate, state clipping, or hidden boundary
repair.  Log coordinates keep every reconstructed physical state `N=exp(x)`
strictly positive.  A finite positive `sigma` solves a viscous surrogate, not
the deterministic first-order HJB itself, so the declared
`0.10 / 0.05 / 0.025` sweep is required and every run is retained.

### Initial bang-bang conditioning

The unchanged official model initializes each component of `Z_0` from
`Uniform[-0.1,0.1]`.  At the initial tumor state,

```text
psi_0 = gamma - phi·Z_0/sigma.
```

When `psi_0 > 0`, the generator term `U min(psi_0,0)` is locally constant in
`Z_0`; its direct control-Hamiltonian gradient is zero.  Terminal matching can
still train `Z_0` through the stochastic `Z_0 dW` term, so this is a
control-generator dead zone, not complete gradient death.  Smaller `sigma`
amplifies `Z/sigma` and changes this conditioning materially.

Every tumor manifest therefore records the exact initial `Z_0`, `phi·Z_0`,
`psi_0`, activation/dead-zone flag, required threshold `gamma*sigma`, and a
clearly labelled Gaussian approximation to the positive-psi probability.  It
also records final positive/negative-psi and endpoint-control fractions.  No
initialization correction or alternate reference SDE is applied in this
track: the official initializer and declared configuration are retained,
and realized `J` is evaluation-only rather than a tuning or selection signal.

For the 21-dimensional nominal parameters, `sum(phi)=16.4558799`, so the
official initializer guarantees `phi·Z_0 <= 1.6455880`.  At `sigma=0.10`, the
activation threshold is `gamma*sigma=2.0`; hence the initial control-generator
dead zone is mathematically guaranteed for every seed.  For `sigma=0.05` and
`0.025`, Gaussian approximations to the positive-psi probability are
`0.9999988` and `0.9908389`, respectively.  These probabilities diagnose
conditioning only and are not used to change the declared sweep.

## Commands

From the repository root:

```bash
.venv/bin/python -m faithful_related_work.deepbsde.runner \
  hjb-lq-smoke \
  --output-dir /tmp/deepbsde_hjb_lq_smoke

# Formal five-seed 100D benchmark; this is substantially more expensive.
.venv/bin/python -m faithful_related_work.deepbsde.runner \
  hjb-lq-full \
  --benchmark-config faithful_related_work/deepbsde/configs/hjb_lq_full_benchmark.json \
  --output-dir paper_runs/faithful_related_work/deepbsde_hjb_lq_full

.venv/bin/python -m faithful_related_work.deepbsde.runner \
  tumor-sweep \
  --smoke \
  --output-dir /tmp/deepbsde_tumor_smoke

.venv/bin/python -m faithful_related_work.deepbsde.runner \
  tumor-sweep \
  --config faithful_related_work/deepbsde/configs/tumor_sweep.json \
  --output-dir paper_runs/faithful_related_work/deepbsde_tumor_full
```

The smoke command preserves all three viscosities but uses one seed, four time
intervals and one optimizer update.  It validates wiring only; it is not an
experimental result.  The full configuration declares three seeds per
viscosity and 2,000 optimizer iterations.

Both full runners refuse to start when their output directory is non-empty.
This prevents accidental mixing or overwriting of run artifacts.

## Artifacts

Every tumor run writes:

- `config.json`: the exact immutable run configuration;
- `training_history.csv`: step, terminal loss, `Y_0`, and elapsed time;
- `model.weights.h5`: all official DeepBSDE trainable variables;
- `canonical_control.npz`: canonical `t` and ZOH interval `u`, plus states,
  gradients, `Z`, switching function, viscosity, and seed;
- `realized_metrics.json`: reevaluation by `tumor_problem.evaluate_zoh_control`;
- `manifest.json`: seed, viscosity, formulation, environment, source hashes,
  artifact hashes, and realized metrics.

The sweep root writes `sweep_manifest.json` and applies no post-hoc model or
seed selection.  Runtime is expected to be substantial for the full 9-run
configuration; only the smoke runs belong in routine tests.

Each HJB-LQ full seed writes `training_history_with_error.csv`, an H5
checkpoint, the exact official solver config, and a manifest.  The full root
manifest reports the five-run mean, sample standard deviation, mean/max
relative error, source/config hashes, and all seed manifests.  A smoke
manifest is permanently marked `wiring_smoke_not_a_numerical_reproduction`.

## Tests

```bash
.venv/bin/python -m unittest tests.test_faithful_deepbsde -v
```

The tests cover the HJB generator formula, tensor shapes, exact Hamiltonian
argmin, absence of clipping, seeded sampler determinism, required viscosity
sweep, official HJB-LQ path, the exact 100D reference value, smoke/full
classification, overwrite refusal, initial dead-zone diagnostics, and an
end-to-end tumor artifact smoke.
