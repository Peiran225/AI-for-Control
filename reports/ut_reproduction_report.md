# Transformer \(u(t)\) Experiment Report

This report reproduces the manuscript's time-dependent Transformer control strategy

$$
u_\theta(t):[0,T]\rightarrow [0,u_{\max}],
$$

trained by PMP/KKT optimality gaps.

## 1. Model and Training Objective

We use the population dynamics

$$
\dot N_i(t)=\big(r_i-\phi_i u(t)-M_iG(N(t))\big)N_i(t),
\tag{1}
$$

$$
G(N)=\log\left(1+\frac{1}{m}\sum_{k=1}^m N_k\right),
\tag{2}
$$

and the cost

$$
J(u)=\alpha^\top N(T)+\int_0^T\big(\beta^\top N(t)+\gamma u(t)\big)\,dt.
\tag{3}
$$

For the reported run, \(T=10\), \(m=21\), \(u_{\max}=3\), \(\alpha=1\), \(\beta=0.1\), \(\gamma=20\), and \(N_i(0)=10\). The control is represented by a small Transformer encoder over normalized time, with

$$
u_\theta(t_k)=u_{\max}\sigma(g_\theta(t_k)).
$$

The training loss follows the manuscript's PMP/KKT optimality-gap formulation. After rolling out \(N_\theta(t)\), we solve the costate equation backward and define

$$
\psi(t)=H_u(N,\lambda,u)
=\gamma-\sum_i\phi_i\lambda_i(t)N_i(t).
\tag{4}
$$

The loss has two optimality-condition components:

| component | role |
|---|---|
| non-singular KKT loss | enforces \(u=0\) when \(\psi>0\), and \(u=u_{\max}\) when \(\psi<0\) |
| singular loss | near \(\psi=0\), matches \(u_\theta(t)\) to the singular candidate \(u_{\mathrm{sing}}(N(t))\) |

The total training loss is

$$
\mathcal L
=
\operatorname{mean}\left[
q(t)(u_\theta(t)-u_{\mathrm{sing}}(t))^2
+(1-q(t))\ell_{\mathrm{KKT}}(t)
\right]
+10^{-4}\operatorname{mean}\big[(u_{k+1}-u_k)^2\big].
$$

Here \(q(t)\) is a smooth weight that switches between the singular condition near \(\psi(t)=0\) and the boundary KKT condition away from \(\psi(t)=0\).

## 2. Learned \(u(t)\) and \(N(t)\)

The learned control starts high, transitions to a lower interior/singular-like region, and increases again near the terminal portion. The state trajectory decreases substantially from the initial population.

![Transformer u(t) and N(t)](../paper_runs/open_loop_ut_report/ut_nt_trajectory_clean.png)

The \(N(u)\) phase plot below shows the same rollout with population plotted directly against the applied control value. Color indicates time.

![N(u) phase plot](../paper_runs/open_loop_ut_report/nu_phase_plot_clean.png)

## 3. Training Loss Trajectories for PMP/KKT Conditions

The table reports the smallest recorded training loss. In this experiment, the training loss is the manuscript's PMP/KKT optimality gap, composed of the singular condition and the non-singular Hamiltonian minimization condition.

| metric | value |
|---|---:|
| total training loss / PMP-KKT optimality gap | 0.02637 |
| singular-condition training loss | 0.00167 |
| non-singular Hamiltonian-minimization training loss | 0.02471 |
| objective \(J\) on the training grid | 384.76 |
| range and mean of \(u_\theta(t)\) | min 1.038, max 2.758, mean 1.372 |
| terminal mean state | 1.207 |

The trajectory separates the two optimality conditions in the manuscript. When \(\psi(t)\approx0\), the singular condition is active; when \(\psi(t)\neq0\), the non-singular Hamiltonian minimization condition pushes the control to the appropriate boundary \(0\) or \(u_{\max}\). The singular-condition gap is small, while the remaining error mainly comes from the non-singular Hamiltonian-minimization condition, especially near switching regions.

![Training loss trajectories](../paper_runs/open_loop_ut_report/training_loss_trajectory_clean.png)

The next plot shows pointwise PMP/KKT diagnostics along the final learned trajectory. The smooth weight \(q(t)\) separates the two regimes: near \(\psi(t)=0\), the plot emphasizes the singular-condition error; away from \(\psi(t)=0\), it emphasizes the boundary KKT error.

![Pointwise PMP/KKT components](../paper_runs/open_loop_ut_report/pmp_condition_components_clean.png)

## 4. Comparison With Gu et al. [3]

Here [3] refers to Gu, Xiong, and Chen, *Pontryagin Optimal Control via Neural Networks* (arXiv:2212.14566). Their Neural-PMP / PMP-gradient method also follows a Pontryagin-style procedure: forward state integration, backward costate recursion, and Hamiltonian-gradient updates of a discrete control sequence. For comparison on the same model and parameters, we implemented the corresponding control-update step from [3].

| method | control update / training criterion | PMP/KKT gap | objective \(J\)[^1] | note |
|---|---|---:|---:|---|
| Transformer \(u_\theta(t)\) | paper PMP/KKT optimality-gap loss | 0.0523 | 386.70 | main \(u(t)\) reproduction |
| Neural-PMP implementation of [3] | Hamiltonian-gradient update of control sequence | 7.47 | 387.02 | related-work comparison |
| direct grid cost minimization | minimize discretized \(J\) | 0.805 | 386.47 | reference only |
| constant \(u=1.5\) | no training | 76.89 | 400.40 | scale check |

Under the same PMP/KKT gap calculation, the Transformer \(u_\theta(t)\) has a much smaller gap than the Neural-PMP implementation of [3]. When \(J\) is recomputed with the same numerical evaluator, the Transformer also has a slightly lower objective value in this run.

[^1]: The \(J\) values in this table are computed after fixing \(u(t)\), reintegrating \(N(t)\) with a finer-step fourth-order Runge-Kutta method, and then applying the manuscript objective definition. This is only to use the same numerical integration accuracy across methods.

![Neural-PMP control and state rollout](../paper_runs/neural_pmp_baseline_beta01/neural_pmp_ut_nt.png)

![Neural-PMP training curve](../paper_runs/neural_pmp_baseline_beta01/neural_pmp_training_curve.png)

![Reference-cost comparison](../paper_runs/neural_pmp_baseline_beta01/neural_pmp_reference_gap_closeup.png)

## 5. Conclusion

The requested \(u(t)\) reproduction is complete. The Transformer control trajectory trained with the manuscript's PMP/KKT optimality-gap loss is smooth and satisfies the control bounds, reduces the training optimality gap from 76.26 to 0.02637, and gives \(J\approx386.70\) under the common numerical evaluation. Compared with the Neural-PMP implementation of Gu et al. [3], the Transformer \(u(t)\) has a lower PMP/KKT gap and a slightly lower objective \(J\) on the nominal \(\beta=0.1\) setting.

The state-dependent extension \(u(t,N)\) is not included in this report because it requires different optimality conditions.
