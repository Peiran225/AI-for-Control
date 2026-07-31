# [6] LyZNet PINN-PI faithful original-method track

This track reproduces the checked-out authors' pendulum PINN policy-iteration
example. It does not create a tumor variant and is intentionally excluded from
the common tumor realized-J ranking.

## What is reproduced

`run_original_pinn_pi.py --full` constructs the author's
`lyznet.ControlAffineSystem` and calls the author's `lyznet.neural_pi` with the
same explicit configuration as
`external/lyznet/examples/icml24-pinn-pi/pendulum/pendulum.py` at revision
`21d8e81a4a21aa990532275836944d9384089f09`:

- dynamics `x1_dot = x2`, `x2_dot = 19.6 sin(x1) - 4 x2 + 40 u`;
- domain `[-2,2]^2`, default `Q = I`, and `R = 2`;
- initial stabilizer `u0 = -(1/40)x1 - (19.6/40)sin(x1)`;
- 10 policy iterations;
- 300,000 collocation points per policy evaluation;
- two hidden layers of width 10;
- learning rate `1e-3`, batch size 32, and 10 epochs per evaluation.

The full branch adds no optimizer, plotting, verification, or stopping
overrides absent from the author example. `--smoke` changes only the budget to
one iteration, one epoch, and a small collocation set. It validates the
original pendulum pipeline but is not a paper-budget result.

## Why there is no tumor mode

The original method assumes a stationary infinite-horizon stabilization
problem with quadratic stage cost:

```text
x_dot = f(x) + g(x)u,
l(x,u) = x^T Q x + u^T R u,
u_(i+1)(x) = -0.5 R^(-1) g(x)^T grad V_i(x).
```

It also assumes an equilibrium at the origin and an admissible stabilizing
initial policy. The common tumor problem is finite-horizon, uses a bounded
scalar control, has a terminal cost, and penalizes drug linearly. Its
Hamiltonian minimizer is bang-bang/singular rather than the unconstrained
quadratic formula above. Changing LyZNet to that problem class is a new
adaptation, not a faithful reproduction. This track therefore produces no
tumor control and must not appear in the tumor realized-J ranking.

## Real dReal requirement

The author implementation uses dReal expressions for symbolic controller
updates and derivatives even when optional final formal verification is not
requested. The runner imports and exercises the compiled dReal API and fails
if it is absent or replaced by a repository-local stub.

The Dockerfile pins the official immutable base image
`dreal/dreal4:latest@sha256:6ec1d55...b57d4`. It is Linux/AMD64 and supplies
its binding for Python 3.6, so the scientific Python stack is also pinned to
compatible versions. On ARM Macs Docker runs it through AMD64 emulation.

Static host preflight (it reports, but does not hide, a missing local dReal):

```bash
python3 faithful_related_work/lyznet/run_original_pinn_pi.py --preflight
```

Build and run a true-dReal one-iteration smoke:

```bash
LYZNET_MODE=--smoke \
  bash faithful_related_work/lyznet/run_in_docker.sh \
  paper_runs/faithful_related_work/lyznet_pendulum_smoke_seed123 \
  --smoke-collocation 64
```

Run the exact author budget:

```bash
bash faithful_related_work/lyznet/run_in_docker.sh \
  paper_runs/faithful_related_work/lyznet_pendulum_seed123
```

The output directory must be new or empty. This prevents upstream
`neural_learner` from silently loading an earlier checkpoint rather than
retraining. Each completed run writes `manifest.json` with:

- upstream revision and tracked-tree cleanliness;
- source and runner SHA-256 hashes;
- seed and exact/effective budget;
- Python, PyTorch, platform, and real-dReal origin;
- every checkpoint's path, size, and SHA-256 hash;
- explicit `tumor_comparable: false` provenance.

## Verified smoke in this workspace

The image built successfully and imported the real binding from
`/usr/local/lib/python3.6/dist-packages/dreal/__init__.py`. Two independent
seed-123 smoke runs each completed one author `neural_pi` iteration and wrote
the same checkpoint hash:

```text
563443b648489583871b5694e0f61cf50c078eb75b117932a6f31fabcec48b95
```

This deterministic smoke validates environment, formula construction,
training, analytic policy improvement, checkpointing, and provenance. It is
not a substitute for the 10 x 300,000 x 10 full run.
