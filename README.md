# Transformer Feedback Control

Clean upload preview for the PMP/KKT optimal-control experiments.

## Method

The main comparison is:

```text
Transformer open-loop baseline      u_theta(t)
Transformer feedback policy         u_theta(t, N)
Refined Transformer feedback        u_theta(t, N), low-lr refinement
```

All three are trained with the PMP/KKT optimality-gap loss. The open-loop baseline is still open-loop because it only observes time `t`; the feedback policies observe both time and the current population state `N`.

The MLP feedback run is kept as an earlier baseline for context.

Here `beta` is the running population-cost weight in the objective. The curated result figures focus on `beta=0.1`, which is the setting where we ran the full seed comparison and Transformer feedback refinement experiments.

## Files

```text
scripts/train_openloop.py      # Transformer open-loop baseline u(t)
scripts/train_feedback.py      # Transformer feedback policy u(t, N)
scripts/refine_feedback.py     # Refined Transformer feedback
scripts/evaluate.py            # policy generalization evaluation
scripts/run_example.sh         # example beta=0.1 commands

results/method_comparison_summary.csv
results/method_comparison_log.png
results/method_comparison_zoom.png
results/improvement_path.png
results/transformer_ablation_summary.csv
results/transformer_ablation.png
```

## Main Result

For beta=0.1, all-scale best PMP/KKT gap:

```text
Transformer open-loop baseline  5.743
MLP feedback baseline           0.963
Transformer feedback policy     0.041
Refined Transformer feedback    0.0187
```

The recommended final policy is `Refined Transformer feedback`.

## Notes

This preview intentionally excludes raw training logs, checkpoints, virtual environments, server-specific scripts, and intermediate experimental variants.
