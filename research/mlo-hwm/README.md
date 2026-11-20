# MLO-HWM: Multi-Level Optimization for Hierarchical World Models

New research project (started 2026-11-20). Target: **NeurIPS 2027**. Official dates are not yet posted; the projected deadline is late May 2027.

**Question.** Hierarchical world models train each level with its own prediction loss, so the objective mismatch of model-based RL compounds across levels. Can we instead learn the hierarchy as a *nested multi-level optimization*, where each level's model is the solution of a sub-problem whose output feeds the level above, and the abstraction is chosen by real decision quality through hypergradients?

| File | Contents |
|---|---|
| [`proposal.md`](proposal.md) | Formulation, hypergradient, planned theory, experiment plan, risks, timeline |
| [`related_work.md`](related_work.md) | Literature map with links and a novelty statement (with verification flags) |
| [`pilot/`](pilot/) | JAX toy pilot: 4-level MLO on an LQ system with planted two-level objective mismatch |

## Reproduce the pilot (CPU, about 15 minutes)

```bash
pip install "jax[cpu]" optax matplotlib
```

```bash
cd research/mlo-hwm/pilot
python mlo_lq_pilot.py --seeds 3 --steps 200 --spsa_pairs 8 --outer_rollouts 64 --out results/budget_big
python mlo_lq_pilot.py --seeds 3 --steps 200 --spsa_pairs 2 --outer_rollouts 8  --out results/budget_mid
python mlo_lq_pilot.py --seeds 3 --steps 200 --spsa_pairs 1 --outer_rollouts 1  --out results/budget_small
python summarize.py
```

## Immediate next steps

1. Read arXiv 2605.15975, 2411.00361, 2604.03208 (HWM) and 2610.06805 (H-JEPA) in full, and confirm novelty.
2. Write the proof of the hierarchical simulation lemma (T1 in the proposal).
3. Replace the pilot's zeroth-order outer gradient with a real-data critic (OMD-style) and re-run the budget sweep. This is the variant that scales.
4. Port to a JAX TD-MPC2-style latent backbone on state-based OGBench antmaze, with distractors. Go/no-go by end of January 2027.
