# External Method Sources

The related-work experiments start from the following public source revisions. The nested repositories are development checkouts; recorded adaptation patches are stored in `external_patches/`, although the isolated Neural-PMP runner and local DeepBSDE tumor equation do not execute every patched external path.

| Method | Source | Revision | Tumor adaptation |
|---|---|---|---|
| DeepBSDE [2] | `https://github.com/frankhan91/DeepBSDE` | `e76fed80b1995daf4a0dd6d3e2cf64a0931dabda` | official solver plus a local 21-dimensional log-state tumor equation, `M=0.5`, and declared `sigma={0.100,0.050,0.025}` sweep |
| Neural-PMP [3] | `https://github.com/ChengyangGU/NeuralPMP2024` | `e8a6269976bfd77ce7f7000ca6ca9ea3b5022833` | paper/released-code-informed isolated reimplementation with a tumor environment, learned-dynamics PMP-gradient path, and multiple control initializations; the released solver is not executed |
| PINN-PI [6] | `https://git.uwaterloo.ca/hybrid-systems-lab/lyznet.git` | `21d8e81a4a21aa990532275836944d9384089f09` | faithful track executes the original pendulum path; the legacy finite-horizon tumor patch is excluded from reproduction claims |
| Adaptive BVP/HJB [7] | `https://github.com/Tenavi/HJB_NN` | `11cfa0e721f02bf36d0cdf03b16e08cf6a06e49c` | tumor problem definition, time-dependent value network, BVP value/costate data, L-BFGS training, NN-warm-start BVP evaluation |
| PI-DeepONet [5] | arXiv source `2406.10920` | v1 source archive | paper-algorithm branch/trunk DeepONet and policy iteration; no author code release was used |

## Recreate the External Checkouts

Clone each released implementation at the revision above, then apply the
corresponding patch from the repository root:

```bash
git clone https://github.com/frankhan91/DeepBSDE external/DeepBSDE
git -C external/DeepBSDE checkout e76fed80b1995daf4a0dd6d3e2cf64a0931dabda
git -C external/DeepBSDE apply ../../external_patches/deepbsde_tumor.patch

git clone https://github.com/Tenavi/HJB_NN external/HJB_NN
git -C external/HJB_NN checkout 11cfa0e721f02bf36d0cdf03b16e08cf6a06e49c
git -C external/HJB_NN apply ../../external_patches/hjb_nn_tumor.patch

git clone https://github.com/ChengyangGU/NeuralPMP2024 external/NeuralPMP2024
git -C external/NeuralPMP2024 checkout e8a6269976bfd77ce7f7000ca6ca9ea3b5022833
git -C external/NeuralPMP2024 apply ../../external_patches/neural_pmp_tumor.patch

git clone https://git.uwaterloo.ca/hybrid-systems-lab/lyznet.git external/lyznet
git -C external/lyznet checkout 21d8e81a4a21aa990532275836944d9384089f09
git -C external/lyznet apply ../../external_patches/pinn_pi_tumor.patch
```

The final command reconstructs a historical tumor-adaptation worktree only; it
does not make that adaptation a faithful [6] reproduction. The report's [6]
fidelity evidence comes exclusively from the original pendulum runner in
`faithful_related_work/lyznet`, which records the real dReal binding and sets
`tumor_comparable: false`.

## HJB_NN original benchmark and tumor continuation

The original satellite path is kept separate from the tumor adaptation.  The
released `t0` checkpoint is evaluated on the released independent 1,000-
trajectory BVP set, and deterministic from-scratch runs use the public
6-64-64-64-1 value/costate architecture, full-batch L-BFGS, and Algorithm 4.1
stopping rule.  TensorFlow 1 `contrib` is unavailable in the current runtime, so
the checkout contains a recorded TensorFlow-2 compatibility port; it is not an
unchanged author tree or an exact replay of the unpublished training seed/data.

The tumor Hamiltonian is affine in the bounded control and has a dominant singular arc. The hard minimizer makes the original `solve_bvp` path singular: repeated attempts exceeded 20,000 mesh nodes before solving the first trajectory. The declared [7] adaptation therefore uses the entropy-smoothed minimizer

```text
u_tau = 3 sigmoid(-psi/tau), tau in {10,5,2}.
```

The declared entropy term is included in the running cost and characteristic
value labels, so this is a regularized tumor OCP rather than a smoothing-only
implementation trick.  Each of three seeds starts from 16 training and 16
trajectory-disjoint validation characteristics at `tol=1e-3`; the uniform full
BVP budget uses 16 horizon-marching stages and up to 100,000 nodes.  The released
time-dependent HJBnet logic then trains `V(t,N)` with value/costate supervision,
full-batch L-BFGS, and Algorithm 4.1 NN-warm-start adaptive sampling.  Every
declared tau/seed run is retained.

## Retained-artifact provenance limits

The retained tumor artifacts do not all prove execution-time identity for the
local adaptation layer. Neural-PMP and PI-DeepONet bind result artifacts but do
not bind their executed local runners. DeepBSDE worker manifests bind the
official external solver/equation sources, but the retained older workers do
not bind the local adaptation files
`faithful_related_work/deepbsde/runner.py` and `equations.py`. The independent
audit records the current hashes of these sources, which can detect subsequent
drift but cannot retroactively prove which local bytes produced the retained
runs. Current runner code records the local DeepBSDE sources for future runs;
that improvement does not upgrade the provenance of the existing artifacts.

## Evaluation Rule

Each method may estimate the tumor cost internally (for example `Y0`, `V_theta(0,N0)`, or a BVP value). The report retains that native estimate as a diagnostic. For HJB-NN the native estimate and BVP labels belong to the entropy-regularized objective.  The primary cross-method number is instead the unregularized realized tumor objective obtained by executing the generated breakpoint-aligned ZOH control under the canonical dynamics in `tumor_problem.py`.
