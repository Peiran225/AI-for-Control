#!/usr/bin/env python3
"""Build the frozen teacher-free experiment summary and three-way plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/teacher_free_n800_20260720"
WINNER = OUT / "curriculum_temp070_learntau_final_low_lr3e5"
GUIDED = (
    ROOT
    / "outputs/time_only_native_n800_transformer_20260720"
    / "optimality_final_pg1e4"
)
DIRECT = ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n800.npz"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    blind = load_json(WINNER / "blind_direct_comparison.json")
    audit = load_json(WINNER / "checkpoint_reload_and_inference.json")
    hessian = load_json(
        WINNER
        / "reduced_full_gradient_hessian/teacher_free_n800/summary.json"
    )
    critical = load_json(
        WINNER
        / "reduced_full_gradient_hessian/teacher_free_n800"
        / "critical_cone_sufficient_check.json"
    )
    guided = load_json(GUIDED / "summary.json")["selected"]
    guided_hessian = load_json(
        GUIDED
        / "reduced_full_gradient_hessian/native_n800_final/summary.json"
    )
    direct = blind["direct_n800"]
    teacher_free = blind["teacher_free"]

    rows = [
        {
            "method": "teacher-free Transformer",
            "J": teacher_free["high_accuracy_J"],
            "projected_gradient_linf": teacher_free["projected_gradient_linf"],
            "projected_gradient_rms": teacher_free["projected_gradient_rms"],
            "plateau_relative_percent": teacher_free["plateau_relative_percent"],
            "early_width_10_90": teacher_free["early_width_10_90"],
            "late_width_10_90": teacher_free["late_width_10_90"],
            "exact_upper_bound_count": teacher_free["exact_upper_bound_count"],
            "kkt_pass_1e4": hessian["kkt_pass_at_tolerance"],
            "critical_span_min_eigenvalue": critical[
                "free_plus_weak_subspace_min_eigenvalue"
            ],
            "teacher_used_in_training": False,
        },
        {
            "method": "direct-guided Transformer",
            "J": guided["J"],
            "projected_gradient_linf": guided["projected_gradient_linf"],
            "projected_gradient_rms": guided["projected_gradient_rms"],
            "plateau_relative_percent": guided["plateau_relative_percent"],
            "early_width_10_90": guided["early_width_10_90"],
            "late_width_10_90": guided["late_width_10_90"],
            "exact_upper_bound_count": guided["exact_upper_bound_count"],
            "kkt_pass_1e4": guided_hessian["kkt_pass_at_tolerance"],
            "critical_span_min_eigenvalue": load_json(
                GUIDED
                / "reduced_full_gradient_hessian/native_n800_final"
                / "critical_cone_sufficient_check.json"
            )["free_plus_weak_subspace_min_eigenvalue"],
            "teacher_used_in_training": True,
        },
        {
            "method": "direct n=800",
            "J": direct["high_accuracy_J"],
            "projected_gradient_linf": direct["projected_gradient_linf"],
            "projected_gradient_rms": direct["projected_gradient_rms"],
            "plateau_relative_percent": direct["plateau_relative_percent"],
            "early_width_10_90": direct["early_width_10_90"],
            "late_width_10_90": direct["late_width_10_90"],
            "exact_upper_bound_count": direct["exact_upper_bound_count"],
            "kkt_pass_1e4": True,
            "critical_span_min_eigenvalue": None,
            "teacher_used_in_training": None,
        },
    ]
    with (OUT / "TEACHER_FREE_FINAL_COMPARISON.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with np.load(WINNER / "solution.npz") as data:
        t = np.asarray(data["t"])
        u_free = np.asarray(data["u"])
    with np.load(GUIDED / "selected_solution.npz") as data:
        u_guided = np.asarray(data["u"])
    with np.load(DIRECT) as data:
        u_direct = np.asarray(data["u"])
    figure, axes = plt.subplots(1, 3, figsize=(10.9, 3.25))
    styles = (
        (u_direct, "direct n=800", "#16837A", 1.55),
        (u_guided, "direct-guided Transformer", "#6F5A9B", 1.45),
        (u_free, "teacher-free Transformer", "#225B8A", 1.5),
    )
    for axis in axes:
        for control, label, color, width in styles:
            axis.step(t, control, where="post", label=label, color=color, lw=width)
        axis.grid(alpha=0.18)
        axis.set_ylim(-0.05, 3.08)
        axis.set_xlabel("time")
    axes[0].set(xlim=(0.0, 10.0), ylabel="control", title="Full horizon")
    axes[1].set(xlim=(0.15, 0.80), title="Early transition")
    axes[2].set(xlim=(8.55, 9.30), title="Late transition")
    axes[0].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(OUT / "teacher_free_guided_direct_comparison.png", dpi=220)
    figure.savefig(OUT / "teacher_free_guided_direct_comparison.pdf")
    plt.close(figure)

    result = {
        "winner": str(WINNER),
        "selection_frozen_before_direct_blind_test": True,
        "teacher_free_training": {
            "direct_solution_or_labels": False,
            "manual_target": False,
            "switching_time_or_mask": False,
            "objective_J_as_training_loss": False,
            "start": "manuscript PMP/KKT-only seed-4 checkpoint",
            "resolution_curriculum": [200, 400, 800],
            "final_method": "detached projected full-gradient fixed-point",
            "full_state_dependence_N_of_u": True,
        },
        "teacher_free_metrics": teacher_free,
        "direct_guided_metrics": guided,
        "direct_n800_metrics": direct,
        "teacher_free_minus_direct_J": blind["teacher_free_minus_direct"][
            "high_accuracy_J"
        ],
        "teacher_free_minus_direct_guided_J": teacher_free["high_accuracy_J"]
        - guided["J"],
        "teacher_free_minus_direct_guided_projected_gradient_linf": teacher_free[
            "projected_gradient_linf"
        ]
        - guided["projected_gradient_linf"],
        "first_order": {
            "tolerance": hessian["kkt_tolerance"],
            "projected_gradient_linf": hessian["projected_kkt_linf"],
            "passes": hessian["kkt_pass_at_tolerance"],
        },
        "second_order": {
            "free_variables": hessian["free_variables"],
            "weak_active_variables": hessian["weakly_active_variables"],
            "strong_active_variables": hessian["strongly_active_variables"],
            "free_hessian_min_eigenvalue": hessian[
                "hessian_min_eigenvalue_free"
            ],
            "free_plus_weak_span_min_eigenvalue": critical[
                "free_plus_weak_subspace_min_eigenvalue"
            ],
            "negative_eigenvalues_on_free_plus_weak_span": critical[
                "free_plus_weak_negative_eigenvalues"
            ],
            "positive_on_tolerance_defined_critical_cone": critical[
                "positive_on_critical_cone_sufficient_check"
            ],
        },
        "checkpoint_reload_linf": audit["checkpoint_reload_control_linf"],
        "loaded_network_inference": audit["inference"],
        "additional_training_wall_seconds_from_existing_manuscript_checkpoint": audit[
            "training_wall_total_seconds"
        ],
        "lineage": audit["lineage"],
        "visual_QA": {
            "passed": True,
            "assessment": (
                "Natural high-low-high trajectory with exact upper-bound arcs; "
                "no full-boundary or constant collapse.  The late transition is "
                "visibly broader than direct n=800."
            ),
        },
        "conclusion": (
            "The teacher-free run reaches the 1e-4 numerical first-order screen "
            "and positive curvature on a superset of the tolerance-defined "
            "critical cone, but it does not outperform direct n=800 or the "
            "direct-guided Transformer in J; its late switch is also broader."
        ),
    }
    (OUT / "TEACHER_FREE_FINAL_SUMMARY.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# Teacher-free n=800 Transformer experiment

## Frozen protocol

The selected model starts from the manuscript PMP/KKT-only seed-4 checkpoint, follows an `n=200 -> 400 -> 800` resolution curriculum, and then uses a detached projected full-gradient fixed-point continuation.  Training reads no direct solution or label, no manual target, and no switching time or mask.  `J` is recorded only as a diagnostic and is not a training loss.  The direct n=800 artifact was first opened after the checkpoint was frozen.

## Result

| method | J | projected-gradient Linf | plateau range (%) | early width | late width |
|---|---:|---:|---:|---:|---:|
| teacher-free Transformer | {teacher_free['high_accuracy_J']:.9f} | {teacher_free['projected_gradient_linf']:.3e} | {teacher_free['plateau_relative_percent']:.4f} | {teacher_free['early_width_10_90']:.5f} | {teacher_free['late_width_10_90']:.5f} |
| direct-guided Transformer | {guided['J']:.9f} | {guided['projected_gradient_linf']:.3e} | {guided['plateau_relative_percent']:.4f} | {guided['early_width_10_90']:.5f} | {guided['late_width_10_90']:.5f} |
| direct n=800 | {direct['high_accuracy_J']:.9f} | {direct['projected_gradient_linf']:.3e} | {direct['plateau_relative_percent']:.4f} | {direct['early_width_10_90']:.5f} | {direct['late_width_10_90']:.5f} |

The teacher-free objective is `{blind['teacher_free_minus_direct']['high_accuracy_J']:.3e}` above direct n=800 (`{blind['teacher_free_minus_direct']['high_accuracy_J_relative_percent']:.3e}%`) and `{teacher_free['high_accuracy_J'] - guided['J']:.3e}` above the direct-guided Transformer.  It therefore does **not** surpass the frozen teacher or direct reference, although the objective gap is small.

## Optimality and audit

- Projected-gradient Linf: `{hessian['projected_kkt_linf']:.3e}`; passes the stated `1e-4` numerical screen.
- Full Hessian includes the complete `N=N(u)` dependence.  The free-plus-weak-active span has minimum eigenvalue `{critical['free_plus_weak_subspace_min_eigenvalue']:.3e}` and no negative eigenvalues.  This span contains the tolerance-defined critical cone.
- Checkpoint reload discrepancy: `{audit['checkpoint_reload_control_linf']:.3e}`.
- Loaded CPU inference for 801 nodes: median `{audit['inference']['median_ms']:.3f} ms` over {audit['inference']['timed_runs']} timed runs with one PyTorch thread.
- Additional curriculum/continuation wall time from the existing manuscript checkpoint: `{audit['training_wall_total_seconds']:.1f} s`.  This excludes the earlier cost of producing that manuscript checkpoint.

## Visual QA

The final trajectory retains the non-degenerate high-low-high structure and exactly reaches `u=3` on {teacher_free['exact_upper_bound_count']} intervals.  It has no constant or nearly all-boundary collapse.  The late transition remains visibly broader than direct n=800, so the teacher-free result should be presented as close but not superior.
"""
    (OUT / "TEACHER_FREE_FINAL_SUMMARY.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
