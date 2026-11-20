"""Aggregate budget-sweep results into a markdown table and a learning-curve figure."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
budgets = [("big", "8 pairs x 64 rollouts"), ("mid", "2 pairs x 8 rollouts"), ("small", "1 pair x 1 rollout")]
methods = ["oracle_lqr_zoh_k", "zero_control", "decoupled_pca", "decoupled_rrr",
           "mlo_spsa_P_only", "mlo_spsa_E_only", "mlo_spsa", "model_free_spsa", "mlo_oracle"]


def fmt(v):
    v = np.asarray(v)
    med = np.median(v)
    return f"{med:.3g} [{v.min():.3g}, {v.max():.3g}]"


lines = ["| method | " + " | ".join(f"{b} ({d})" for b, d in budgets) + " |",
         "|---|" + "---|" * len(budgets)]
data = {}
for b, _ in budgets:
    p = os.path.join(here, "results", f"budget_{b}", "results.json")
    if os.path.exists(p):
        data[b] = json.load(open(p))
for m in methods:
    row = [m]
    for b, _ in budgets:
        row.append(fmt([r[m] for r in data[b]["runs"]]) if b in data else "n/a")
    lines.append("| " + " | ".join(row) + " |")
extra = []
for b in data:
    r0 = data[b]["runs"][0]
    extra.append(f"- `{b}`: {len(data[b]['runs'])} seeds; real env steps per outer step = "
                 f"{r0['real_steps_per_outer_step']}; exploration data = {r0['exploration_data_steps']} steps; "
                 f"outer steps = {data[b]['args']['steps']}")
    extra.append(f"  - relevance of E row-space (1 = fully inside cost-relevant subspace): "
                 f"RRR init {np.mean([r['rrr_E_relevance'] for r in data[b]['runs']]):.2f} -> "
                 f"MLO-SPSA {np.mean([r['mlo_spsa_E_relevance'] for r in data[b]['runs']]):.2f}; "
                 f"alignment of P E with unstable modes: RRR {np.mean([r['rrr_PE_unstable_alignment'] for r in data[b]['runs']]):.2f} -> "
                 f"MLO-SPSA {np.mean([r['mlo_spsa_PE_unstable_alignment'] for r in data[b]['runs']]):.2f}")
table = "\n".join(lines + [""] + extra)
open(os.path.join(here, "results", "summary.md"), "w").write(table + "\n")
print(table)

fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.4), sharey=True)
axes = np.atleast_1d(axes)
colors = {"mlo_spsa": "#1f6feb", "mlo_oracle": "#8250df", "model_free_spsa": "#d1242f",
          "mlo_spsa_E_only": "#57ab5a", "mlo_spsa_P_only": "#bf8700"}
for ax, (b, desc) in zip(axes, [x for x in budgets if x[0] in data]):
    for m, c in colors.items():
        curves = np.array([[v for _, v in r[f"{m}_curve"]] for r in data[b]["runs"]])
        xs = [i for i, _ in data[b]["runs"][0][f"{m}_curve"]]
        ax.plot(xs, np.median(curves, 0), color=c, label=m)
        ax.fill_between(xs, curves.min(0), curves.max(0), color=c, alpha=0.12)
    ax.axhline(np.median([r["oracle_lqr_zoh_k"] for r in data[b]["runs"]]), color="k", ls="--", lw=1, label="true-model LQR (ZOH k)")
    ax.set_yscale("log"); ax.set_title(desc); ax.set_xlabel("outer steps")
axes[0].set_ylabel("real closed-loop cost (median, min-max)")
axes[-1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(here, "results", "curves.png"), dpi=150)
