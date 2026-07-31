# Fidelity contract for related-work tracks

This directory separates two scientifically different claims.

1. **Original-method reproduction** runs the authors' algorithm on an original
   paper benchmark (or an equivalent included benchmark) without changing the
   problem class or the algorithmic update.
2. **Tumor adaptation** keeps the original method's core numerical mechanism
   but replaces the problem definition with the common tumor problem. Every
   necessary change is recorded in a machine-readable manifest. A tumor
   adaptation is never labelled an original-paper reproduction.

All new runners must:

- leave the checked-out author repositories unchanged;
- record the upstream revision, command, environment, seed, training budget,
  selection rule, and hashes of checkpoints and output controls;
- save all candidates, not only the candidate with the best nominal realized
  objective;
- keep native training quantities separate from the common realized objective;
- export a canonical `t, u` artifact evaluated as breakpoint-aligned ZOH by
  `tumor_problem.evaluate_zoh_control`;
- use a validation/stopping rule declared before training rather than tuning on the nominal
  realized objective;
- run at least three independent seeds for a reported tumor aggregate (five is
  the default target where the original paper reports five runs).

## Method-specific invariants

### [2] Deep BSDE

- Preserve the Euler forward SDE/BSDE, one trainable `Z` approximation per time
  slice, terminal matching loss, and Adam training used by the author code.
- Do not silently clip stochastic states. Positivity must be handled by a
  mathematically stated coordinate transformation or boundary condition.
- A vanishing-viscosity claim requires a declared diffusion sweep and a
  convergence study; a single nonzero diffusion is a fixed-diffusion surrogate.

### [3] Neural-PMP

- Learn the one-step dynamics from a seeded state-control data set.
- Follow Algorithm 1: forward state rollout, terminal/backward costates, update
  with the raw Hamiltonian gradient, then project the *action* to its bounds.
- A shared learned-dynamics checkpoint is used across starts for a given seed.
- Do not clip the Hamiltonian gradient to action bounds.

### [5] PI-DeepONet

- Preserve the branch/trunk operator architecture and terminal-function family.
- Use the paper's finite-difference spatial gradient, discrete Laplacian, and
  artificial viscosity `N h`, with the monotonicity condition on `N` checked.
- Perform the outer policy iterations and the exact Hamiltonian argmin.
- Do not sample from a direct optimal trajectory or any nominal reference.

### [6] LyZNet PINN-PI

- Original reproduction uses the author `ControlAffineSystem` and
  `lyznet.neural_pi` path with an admissible stabilizing initial controller,
  quadratic state/control cost, policy evaluation, policy improvement, and the
  paper's local-stability term/verification path.
- The paper studies stationary infinite-horizon stabilization. The finite-
  horizon, bounded, linearly penalized tumor problem is outside that problem
  class. Any tumor experiment must therefore be labelled a non-comparable
  adaptation unless the common problem itself is changed and reported.

### [7] Adaptive HJB-NN

- Preserve BVP characteristic data, joint value/costate supervision, independent
  trajectory-level train/validation sets, full-batch L-BFGS training, and the
  adaptive data-generation/model-refinement rounds in Algorithm 4.1.
- A smoothed Hamiltonian minimizer must be derived from, and included in, the
  same running cost used for the characteristic value labels.
- If smoothing is used to approximate a bounded affine minimizer, report a
  continuation study and distinguish the regularized native value from the
  unregularized common realized objective.

## Common comparison rule

The common evaluator supports only the following statement:

> The exported feasible control realizes the reported objective when executed
> as a declared ZOH control under the common deterministic tumor dynamics.

It does not by itself make training budgets, stochastic surrogates, policy
representations, problem classes, or algorithmic fidelity comparable.
