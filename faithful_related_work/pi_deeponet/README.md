# PI-DeepONet paper-derived implementation ([5])

This package implements Lee and Kim, *Hamilton-Jacobi Based Policy-Iteration
via Deep Operator Learning* (arXiv:2406.10920), Eq. (2.3)--(2.6) and Algorithm
1.  It deliberately does not read any direct-control trajectory.

## Preserved algorithmic invariants

- The value operator is a branch/trunk DeepONet.  The branch receives samples
  of a family of terminal functions; the trunk receives `(t,x)`, and their
  outputs are combined by the unnormalized inner product in paper Eq. (2.6).
- Spatial derivatives are the paper's central finite differences
  `nabla^h V(x)` computed from `V(x+h e_i)` and `V(x-h e_i)`.  Spatial
  autograd is not used.
- The discrete Laplacian is the paper's central `Delta^h`, and the policy
  evaluation residual contains the full artificial-viscosity term
  `N h Delta^h V`.
- Every run checks `N >= max(1, ||f||_inf/2)` using a conservative dynamics
  bound on the declared state/control box.
- Each outer policy is frozen while the next linear policy-evaluation PDE is
  trained.  Policy improvement is the exact pointwise Hamiltonian argmin.
- In every Adam step, the PDE and terminal losses use every declared training
  terminal function, as required by Algorithm 1 lines 5--7.  Randomness is in
  the collocation locations, not in omitting terminal functions from a step.
- The affine bounded tumor Hamiltonian selects exactly `0` or `umax`.  At
  `psi == 0`, where the minimizer is non-unique, the deterministic tie rule is
  the box midpoint and the event is saved in the NPZ artifact.

The paper's Assumption A2 requires a unique Lipschitz Hamiltonian argmin. A
linearly penalized bounded tumor control is bang-bang and violates both the
uniqueness requirement on the switching surface and the associated Lipschitz
condition. The explicit tie rule makes the adaptation executable without
pretending that A2 holds, but the paper's policy-convergence result therefore
does not transfer to this tumor problem.

## Runs

Original-paper 5-state/3-control compact LQR smoke (paper `h=0.005`, `M=3`,
terminal family `g_k(x)=0.3+0.1k||x||^2`, `k=1,2,3`, and held-out
`g(x)=0.57||x||^2`):

```bash
python faithful_related_work/pi_deeponet/run_lqr_smoke.py \
  --output-dir /tmp/pi_deeponet_lqr_smoke
```

The default 20 Adam steps per outer iteration are only a wiring smoke, not a
converged reproduction of the paper's plots.

Algorithm 1 states Adam, but neither the paper nor its arXiv source archive
provides author code.  The paper does not report branch/trunk sizes,
activation/initialization/input scaling, latent dimension, sensor
count/locations, state/time sampling box and distribution, batch sizes, Adam
learning rate or moment-state handling between policy iterations, number of
Adam steps/convergence tolerance, or loss weights.  The runner records every
such choice and labels this as an
implementation of the paper-specified algorithm on the original benchmark,
not an exact numerical reproduction of the published figures.  Gradient
clipping is off by default because the paper does not report it; enabling it is
explicitly an ablation.

There is also an internal typesetting inconsistency in Algorithm 1: its
displayed `L1` omits the running cost `L`, puts the sample sum inside the outer
square, and writes `Adam(-alpha1 L1-alpha2 L2)`.  Those choices conflict with
Eq. (2.3), the surrounding description of a residual loss, and ordinary loss
minimization.  This implementation therefore minimizes the positive mean of
the squared pointwise Eq. (2.3) residuals, including `L`, and records that
interpretive choice in every manifest.  Without author code this ambiguity
cannot be resolved empirically.

Three-seed tumor adaptation:

```bash
python faithful_related_work/pi_deeponet/run_tumor.py \
  --seeds 0,1,2 \
  --output-dir /tmp/pi_deeponet_tumor
```

For the tumor adaptation the network coordinate is `x=N/220` on the declared
box `[0,1]^21`; all objective reporting is converted to physical `N` and
reevaluated by the canonical breakpoint-aligned ZOH evaluator.

Each seed directory contains every outer checkpoint, every corresponding
control candidate, `history.csv`, and final canonical `solution.npz` with
`t,u`.  The run root also contains `summary.csv`.  `manifest.json` records the command, environment, seeds, budgets,
selection rule, paper hash, artifact hashes, numerical invariants, and that no
direct reference was used.  The selected candidate is always the final
declared outer iteration; nominal realized `J` is never used for tuning. No
comparable native scalar is exported; tables label this cell “not exported;
final declared outer selected.”

The independent audit re-executes all nine seed/outer control candidates.
Mean common `J` worsens from `410.244` at outer 0 to `482.694` at outer 1 and
`521.989` at the selected outer 2; every seed's final candidate is worse than
its outer-0 candidate. This run therefore provides no empirical policy-
improvement or convergence evidence.

The tumor network declares normalized coordinates `x=N/220` on `[0,1]^21`,
but the `h=0.02` central stencil requests values outside that box at
`181/182/183` of 200 selected-trajectory time points for seeds 0/1/2. The
reported controls consequently rely heavily on neural-network extrapolation
outside the declared training domain; checking only the stencil center does
not detect this issue.
