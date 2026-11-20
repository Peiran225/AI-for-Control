# Related work map: multi-level optimization for hierarchical world models

Status legend:
- **[S]** found and skimmed through web search in this session (Nov 2026). Check the venue and bibliographic details before citing.
- **[M]** cited from memory. Verify every detail before it goes into the paper.

## 1. Hierarchical / temporally abstract world models (closest competitors)

| Work | What it does | Gap relative to our idea |
|---|---|---|
| **HWM**: Zhang et al., *Hierarchical Planning with Latent World Models*, arXiv [2604.03208](https://arxiv.org/abs/2604.03208) (Apr 2026) [S] | Two latent world models at different time resolutions; the high level is conditioned on latent macro-actions. Moderate temporal abstraction helps long-horizon planning. | Levels share one latent space and are trained with prediction losses. Nothing makes the high-level model *decision-consistent* with the low-level planner. |
| **H-JEPA**: Zhang, Terver, Rabbat, LeCun, Balestriero, *End-to-End Learning of Hierarchical World Models for Visual Planning*, arXiv [2610.06805](https://arxiv.org/abs/2610.06805) (Oct 2026) [S] | End-to-end hierarchy of action-conditioned JEPAs. Each level predicts at a coarser stride in its own latent space, and upper-level predictions become subgoals. Visual AntMaze success rises from 18% to 73% (as reported). | Purely self-supervised per level, with no reward or decision signal. The authors note that an upper-level cost need not improve planning. **This is our main motivation and main baseline.** |
| **AgentOWL**: Piriyakulkij et al., *Joint Learning of Hierarchical Neural Options and Abstract World Model*, arXiv [2602.02799](https://arxiv.org/abs/2602.02799) (2026) [S] | Jointly learns options and an abstract world model over states and time; evaluated on object-centric Atari. | Joint single-level objective, not nested. Check venue. |
| **THICK**: Gumbsch et al., ICLR 2024, [poster](https://iclr.cc/virtual/2024/poster/18558) [S] | Adaptive temporal abstractions from sparse latent context updates. | Abstraction comes from a sparsity prior, not from decision quality. |
| **Director**: Hafner et al., *Deep Hierarchical Planning from Pixels*, NeurIPS 2022 [M] | Manager/worker trained inside a DreamerV2 world model. | Flat world model; the hierarchy lives only in the policy. |
| **Puppeteer**: Hansen et al., *Hierarchical World Models as Visual Whole-Body Humanoid Controllers*, ICLR 2025 [M] | Two TD-MPC2 agents (high level and low level). | Levels trained separately. |
| Shaj, *Learning World Models With Hierarchical Temporal Abstractions: A Probabilistic Perspective*, arXiv [2404.16078](https://arxiv.org/abs/2404.16078) [S] | Probabilistic multi-timescale state-space models. | Prediction-centric. |
| *Exploring the limits of hierarchical world models in RL*, Sci. Reports 2024, [link](https://www.nature.com/articles/s41598-024-76719-w) [S] | Reports where hierarchical world models fail. | Useful negative evidence for motivation. |
| Kobayashi et al., *Emergent temporal abstractions in autoregressive models enable hierarchical RL*, ICLR 2026 [S] | "Internal RL": a controller acts on the internal abstractions of an autoregressive model. | Abstractions emerge rather than being optimized for control. |

## 2. Decision-aware / objective-mismatch model learning (flat, single level)

| Work | Relevance |
|---|---|
| Lambert et al., *Objective Mismatch in MBRL*, L4DC 2020, [arXiv 2002.04523](https://arxiv.org/abs/2002.04523) [S] | Defines the problem. |
| Grimm et al., *The Value Equivalence Principle*, NeurIPS 2020, [arXiv 2011.03506](https://arxiv.org/abs/2011.03506) [S] | A theoretical lens we extend *across levels* (hierarchical value equivalence). |
| Nikishin et al., *Control-Oriented MBRL with Implicit Differentiation* (OMD), AAAI 2022, [arXiv 2106.03273](https://arxiv.org/abs/2106.03273) [S] | Bilevel, using the IFT through the optimal Q. **Our method generalizes this to a nested hierarchy.** Primary flat baseline. |
| Eysenbach et al., *Mismatched No More*, NeurIPS 2022, [arXiv 2110.02758](https://arxiv.org/abs/2110.02758) [S] | Joint model/policy lower bound. Flat baseline. |
| Voelcker et al., *VaGraM*, ICLR 2022 [M]; *λ-AC / λ-models*, [arXiv 2306.17366](https://arxiv.org/abs/2306.17366) [S]; IterVAML [M] | Value-aware losses. The authors report that such losses historically *underperform* in continuous control, a risk we must address. |
| Wei et al., *A Unified View on Solving Objective Mismatch in MBRL*, [arXiv 2310.06253](https://arxiv.org/abs/2310.06253) [S] | Taxonomy: distribution correction, control-as-inference, value equivalence and differentiable planning. Use it to position the work. |
| *Not All Errors Matter: Decision-Relevant Prediction Error Predicts Planning Quality*, [arXiv 2609.32322](https://arxiv.org/abs/2609.32322) (2026) [S] | Gives an evaluation metric we can reuse per level. |
| *How Should World Models Be Evaluated for Embodied Decision-Making?*, [arXiv 2606.15032](https://arxiv.org/abs/2606.15032) (2026) [S] | Position paper that supports a decision-centric evaluation. |

## 3. Multi-level / bilevel optimization machinery

| Work | Use |
|---|---|
| Choe et al., **Betty**: autodiff library for MLO, ICLR 2023, [arXiv 2207.02849](https://arxiv.org/abs/2207.02849) [S] | Dataflow-graph formulation of MLO and a reference implementation. |
| Choe et al., **SAMA**: *Making Scalable Meta Learning Practical*, NeurIPS 2023, [arXiv 2310.05674](https://arxiv.org/abs/2310.05674) [S] | First-order hypergradient with adaptive inner optimizers. Our scalable approximation. |
| Sato, Tanaka, Takeda, *A Gradient Method for Multilevel Optimization*, NeurIPS 2021 [M] | Convergence theory for more than two levels. Starting point for our theorem. |
| Petrulionyte, Mairal, Arbel, *Functional Bilevel Optimization for ML*, NeurIPS 2024, [arXiv 2403.20233](https://arxiv.org/abs/2403.20233) [S] | Notes that MBRL as bilevel is more robust to misspecification. Useful for function-space analysis. |
| Prakash et al., *Bilevel Policy Optimization with Nyström Hypergradients* (BLPO), [arXiv 2505.11714](https://arxiv.org/abs/2505.11714) [S] | Actor-critic as bilevel; stable inverse-HVP. |
| Saxena, Gaur, Aggarwal, *Hypergradient-based Bilevel RL with Improved Sample Complexity*, [arXiv 2607.28849](https://arxiv.org/abs/2607.28849) (2026) [S] | Sample-complexity tools for RL lower levels. |
| *Reward-Estimated Hypergradient for Bilevel RL*, [arXiv 2611.11998](https://arxiv.org/abs/2611.11998) (Nov 2026) [S] | Black-box follower hypergradients, relevant to our zeroth-order outer gradient. |

## 4. Bilevel *planning* with world models (different meaning of "bilevel")

- *Synthesizing world models for bilevel planning* (TheoryCoder), [arXiv 2503.20124](https://arxiv.org/abs/2503.20124) [S]: a symbolic high level with a low-level search. "Bilevel" here means planning levels, not nested optimization. Must be distinguished in the paper.
- *Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning*, [arXiv 2605.15975](https://arxiv.org/abs/2605.15975) [S]: not yet read. **Read it before claiming novelty.**
- *Direct Preference Optimization for Primitive-Enabled Hierarchical RL: A Bilevel Approach*, ICLR 2026, [arXiv 2411.00361](https://arxiv.org/abs/2411.00361) [S]: bilevel *HRL policy* (model-free). Closest "bilevel + hierarchy" work, but it has no world model.

## Novelty statement (provisional)

I found no work that **(i)** trains a *hierarchy* of world models in which **(ii)** each level's model is fit as the solution of a lower-level problem whose output feeds the level above, with **(iii)** abstraction and stride chosen by an outer decision objective through hypergradients. Before submission, run a targeted search of the NeurIPS 2026, ICLR 2027 and ICML 2027 accepted lists and arXiv for "hierarchical" plus "decision-aware", "value-equivalent", "bilevel" or "multilevel" plus "world model".
