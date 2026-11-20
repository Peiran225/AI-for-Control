# Decision-Consistent Hierarchical World Models via Multi-Level Optimization

*Working title. Target venue: NeurIPS 2027, main track. Status: idea plus toy pilot (Nov 2026).*

---

## 0. One-paragraph pitch

Hierarchical world models (HWM, H-JEPA, THICK, Director, Puppeteer, AgentOWL) give large gains on long-horizon planning. Each level, however, is trained with its *own* local loss: reconstruction, latent prediction, or a sparsity prior. The well-known **objective mismatch** of model-based RL (a model that is accurate on average need not be good for control) therefore *compounds across levels*. The high-level model is fit to predict whatever the abstraction exposes, and the abstraction is fit to whatever is easy to predict, so neither is fit to what the planner one level up needs. The H-JEPA authors observe exactly this: an upper-level cost need not improve planning.

We propose to treat the hierarchy as a **nested multi-level optimization (MLO) problem**. Each level's world model is the *solution* of a lower-level problem whose output feeds the level above. The abstraction (state encoder and temporal stride) sits at the top and is selected by the *real* decision objective. Hypergradients flow down the hierarchy through level-wise implicit differentiation. The result is a hierarchy that is *decision-consistent by construction*, with standard flat decision-aware MBRL (OMD, MnM) as the one-level special case.

## 1. Problem statement

Real MDP $\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma)$, with a dataset $\mathcal D$ of real transitions that grows during training.

Hierarchy with $L$ model levels. Level $\ell$ has:
- an abstraction $h^\ell = g^\ell_{\phi}(h^{\ell-1})$, with $h^0=s$ or the encoder output, and a temporal stride $k_\ell$ (possibly learned);
- an abstract action $a^\ell$: a subgoal or latent macro-action; $a^0$ is the primitive action;
- a world model $f^\ell_{\theta_\ell}: (h^\ell_t, a^\ell_t) \mapsto h^\ell_{t+k_\ell}$, plus reward/cost and termination heads.

A hierarchical planner or policy $\pi_\psi=(\pi^L,\dots,\pi^0)$ acts in the stack: level $\ell$ emits $a^\ell$, which becomes the goal of level $\ell-1$.

### 1.1 Multi-level formulation (innermost first)

$$
\begin{aligned}
\textbf{(L1)}\quad & \theta_1^\star(\phi) = \arg\min_{\theta_1}\; \mathbb E_{\mathcal D}\big[\,w_\phi(s,a)\, d\big(f^1_{\theta_1}(h^1_t,a_t),\, h^1_{t+1}\big)\big] \\
\textbf{(L2)}\quad & \theta_2^\star(\phi) = \arg\min_{\theta_2}\; \underbrace{\mathbb E\big[d\big(f^2_{\theta_2}(h^2_t,a^2),\, g^2_\phi(\hat h^1_{t+k_2})\big)\big]}_{\text{consistency with L1 imagination under } \pi^1 \text{ tracking } a^2} \;+\;\lambda\,\underbrace{\mathbb E_{\mathcal D}\big[d(f^2_{\theta_2}(h^2_t,\bar a), h^2_{t+k_2})\big]}_{\text{real }k\text{-step anchor}} \\
&\;\;\vdots \qquad\text{(each level fit to the imagination of the level below, anchored to real data)}\\
\textbf{(L}_{L+1}\textbf{)}\quad & \psi^\star(\phi) = \arg\max_\psi\; J_{\text{model}}\big(\psi;\,\theta^\star_{1:L}(\phi)\big) \quad\text{(planning / actor-critic in the hierarchical imagination)}\\
\textbf{(Outer)}\quad & \min_{\phi}\; \mathcal L_{\text{out}}(\phi) = -\widehat J_{\text{real}}\big(\pi_{\psi^\star(\phi)}\big)
\end{aligned}
$$

The outer variables $\phi$ include the abstraction maps $g^\ell_\phi$, the per-sample/per-dimension weights $w_\phi$ of the model losses, and (relaxed) strides $k_\ell$. $\widehat J_{\text{real}}$ is estimated **only from real data**:
- (a) a critic trained on real transitions (as in OMD); or
- (b) a real-data value-equivalence residual of the high-level value function; or
- (c) a zeroth-order estimate from real rollouts (used in the pilot).

Special cases:
- $L=1$ without the outer weights is OMD-style bilevel MBRL.
- Dropping the nesting and summing all losses gives the "end-to-end joint" training of H-JEPA / AgentOWL.
- Freezing $\phi$ gives decoupled HWM-style training.

### 1.2 Hypergradient

Write $x_1=\theta_1, \dots, x_L=\theta_L, x_{L+1}=\psi$, where $x_j^\star$ solves $\nabla_{x_j} \ell_j(x_j; x_{<j}, \phi)=0$. Differentiating gives a **block lower-triangular** linear system. The reverse-mode hypergradient is

$$
\frac{d\mathcal L_{\text{out}}}{d\phi} = \partial_\phi \mathcal L_{\text{out}} - \sum_{j} \big(\partial^2_{\phi x_j}\ell_j\big)^\top v_j,
\qquad
H_j v_j = \nabla_{x_j}\mathcal L_{\text{out}} - \sum_{i>j}\big(\partial^2_{x_j x_i}\ell_i\big)^\top v_i ,
$$

with $H_j=\partial^2_{x_jx_j}\ell_j$. The adjoints are solved **top-down** ($v_{L+1}, v_L, \dots, v_1$). Each solve is one inverse-Hessian-vector problem, handled by CG, a truncated Neumann series, Nyström, or a SAMA-style first-order / optimizer-aware approximation. Total cost is $L+1$ approximate linear solves per outer step, the same order as Betty's MLO dataflow graph.

## 2. Contributions (planned)

1. **Formulation.** Hierarchical world-model learning as nested MLO. Prior decoupled, end-to-end and flat decision-aware methods are special cases.
2. **Theory.**
   - **(T1) Hierarchical simulation lemma.** The real-return gap of a hierarchical planner is bounded by a sum of *level-wise decision-weighted* errors (value-equivalence residuals at each level's stride) rather than raw prediction errors. A level can be prediction-accurate and still be decision-useless, and conversely. This motivates nesting.
   - **(T2) Hypergradient error bound** for $L$ nested levels with inexact inner solves. Error propagates through the triangular adjoint chain. The goal is an explicit dependence on level-wise conditioning $\kappa_j$ and inner-solve accuracy $\epsilon_j$; we expect at most a product $\prod_j \kappa_j$, and want to show it is avoidable with warm starts.
   - **(T3) Convergence** of the stochastic MLO algorithm to an $\epsilon$-stationary point of $\mathcal L_{\text{out}}$ under strong convexity / PL of the inner levels. This extends Sato et al. (NeurIPS 2021) and the two-level bilevel RL results.
   
   *All three are conjectures to be proved, not results.*
3. **Algorithm: MLO-HWM.** A scalable, single-loop implementation with warm-started inner solvers, top-down adjoint propagation and first-order approximations. Two plug-in backbones: TD-MPC2-style latent models and JEPA-style hierarchies (HWM / H-JEPA).
4. **Experiments** showing that nesting beats decoupled and end-to-end hierarchical training on long-horizon control, with a diagnosis of *which level's* mismatch is fixed.

## 3. Pilot evidence (toy, CPU, Nov 2026)

See [`pilot/README.md`](pilot/README.md) for the full setup and numbers. Summary:

- **Setup.** A linear-quadratic system (12-dimensional state, 2 controls) with mismatch planted at **both** levels:
  - 8 large-variance, cost-irrelevant distractor dimensions, which trap level-1 abstractions that are good at prediction or reconstruction;
  - inside the 4 relevant dimensions, 2 high-variance but stable modes versus 2 low-variance, *unstable* modes, which trap the level-2 abstraction.
  
  The hierarchy is: L1 latent model (d=4, ridge LS on real data); L2 stride-4 model fit to L1 imagination plus a real-data cost model; L3 LQR in the abstract space; outer loop over the abstraction (E, P).
- **Result.** RESULTS_PLACEHOLDER

## 4. Experimental plan for the paper

**Benchmarks** (long-horizon, where hierarchy is known to matter):
- OGBench (antmaze-large/giant navigate, cube-double/triple), state and visual;
- Visual AntMaze, Push-T and FourRoom-Distractors, to match the H-JEPA / HWM suites directly;
- DMC humanoid / Puppeteer-style whole-body control, for a continuous-control MBRL comparison;
- *distractor variants* (natural-video backgrounds, irrelevant moving objects), the regime where objective mismatch is known to bite.

**Baselines.**
1. Flat MBRL: DreamerV3, TD-MPC2.
2. Flat decision-aware: OMD, MnM, λ-models / IterVAML.
3. Hierarchical with decoupled training: Director, HWM, Puppeteer.
4. Hierarchical, end-to-end joint: H-JEPA, AgentOWL, THICK.
5. Model-free: HIQL / goal-conditioned IQL on OGBench, plus *equal-real-sample* zeroth-order or policy-gradient baselines. The pilot showed this is a necessary control.

**Ablations.**
- Which levels receive hypergradients: none, top only, bottom only, all. The pilot shows this matters.
- Hypergradient estimator: unrolled, CG, Neumann, SAMA first-order.
- Outer objective: real critic, value-equivalence residual, zeroth-order.
- Learned versus fixed stride.
- Number of levels (2 vs 3).
- Inner-solve budget.

**Metrics.**
- Success/return versus real environment steps (sample efficiency is the claim to defend).
- Planning compute.
- Per-level decision-relevant prediction error (*Not All Errors Matter*, arXiv 2609.32322).
- Wall-clock overhead of the MLO.

**Compute estimate.** Roughly 2–3× the backbone's cost per update for L+1 HVP solves. Budget about 4×A100 for 6 weeks for the main table plus ablations, to be refined after the first OGBench run.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Model-free with the same real budget is competitive** (it was in the pilot at large budgets) | Make sample efficiency the headline metric. Use high-dimensional observations, where model-free gain search is hopeless. Report budget sweeps honestly. |
| Hypergradients are unstable through RL inner levels (non-convex, non-stationary) | Warm-started single-loop inner solvers, damped Neumann / Nyström, gradient clipping on adjoints. Use the planner as L3 (MPC with a differentiable value) rather than an actor when possible. |
| Value-aware losses have historically underperformed in continuous control (Voelcker et al.) | Keep a real-data prediction *anchor* at every level (λ term), so the MLO *reweights* rather than replaces prediction. Outer weights $w_\phi$ start at uniform. |
| Representation collapse when abstractions are decision-driven | The anchor terms plus VICReg/SIGReg-style regularizers already used in JEPA backbones. |
| Overhead does not scale to visual hierarchies | SAMA-style first-order hypergradients. Update $\phi$ every N inner steps. Restrict $\phi$ to the abstraction heads and loss weights, not the full encoder. |
| Novelty overlap with a concurrent 2026–27 paper | Read arXiv 2605.15975 and 2411.00361 in full. Re-run the novelty search monthly (see `related_work.md`). |

## 6. Optional domain study: tumor-control repo

The repo's structured-population tumor model (`utils.py`: 21 trait classes, a single drug, a singular-arc control $u=s(N)$) has a natural two-level hierarchy: trait-resolved dynamics, and an aggregate burden / low-order moments at a coarser time scale. The question of which aggregate to model for *control* is exactly the outer problem here. This could be a small applied section or appendix ("decision-consistent reduced-order models for adaptive therapy") and connects the paper to the existing codebase.

## 7. Timeline (NeurIPS 2027)

The official NeurIPS 2027 dates are **not yet published**. A third-party tracker *projects* abstract registration around May 14, 2027 and the full paper around May 21, 2027. Re-check neurips.cc.

| Window | Milestone |
|---|---|
| Dec 2026 | Read the closest papers in full; finish the novelty check. Prove T1 and draft T2. Port the pilot MLO engine to a JAX TD-MPC2-style backbone. |
| Jan 2027 | State-based OGBench + distractor-DMC: MLO-HWM vs decoupled vs end-to-end (2 levels). Go/no-go checkpoint on a ≥ +10% success or sample-efficiency gain. |
| Feb 2027 | Visual suite (HWM / H-JEPA environments). Add 3-level hierarchies and the learned stride. |
| Mar 2027 | Full baselines and ablations; T3 proof. |
| Apr 2027 | Seeds and error bars (≥5 seeds, 95% CIs); writing; internal review. |
| May 2027 | Abstract registration; submission; code release prepared for the supplement. |
