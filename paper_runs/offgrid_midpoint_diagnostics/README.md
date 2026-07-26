# Frozen-model off-grid midpoint diagnostic

This diagnostic addresses whether the trained `n=800` policies retain their
switching-function behavior at times that were not present in the training
grid.

The network parameters and the original 801-token Transformer context are
frozen. The evaluator then directly queries all 800 interval midpoints,
reintegrates the state and PMP costate under the resulting continuous policy,
and computes the scalar quantities

- `H_u(t)`,
- `d H_u(t) / dt`, and
- `d^2 H_u(t) / dt^2`.

Training-grid coordinates and off-grid midpoint queries have RMS values that
differ by at most 0.542% over `1 <= t < 8.2`. Thus, the midpoint queries do not
show additional degradation relative to the training coordinates. The
absolute residuals remain nonzero, however, so this run does not establish
that all three singular equalities hold throughout the continuous interval.

## Files

- `offgrid_switching_summary.csv`: RMS, mean-absolute, and maximum-absolute
  diagnostics for each policy, state, sampling subset, and interval.
- `offgrid_policy_switching_interior.pdf`: comparison of training-grid and
  off-grid midpoint queries on the interior interval.

## Reproduction

Run:

```bash
python scripts/generate_offgrid_policy_switching_diagnostics.py \
  --time-checkpoint PATH_TO_TIME_ONLY_CHECKPOINT \
  --cf-checkpoint PATH_TO_CF_CHECKPOINT \
  --der-checkpoint PATH_TO_DER_CHECKPOINT
```

The default evaluation has 1,601 coordinates: 801 training-grid nodes and 800
strictly off-grid interval midpoints. Checkpoints are supplied explicitly so
the diagnostic does not depend on machine-specific paths.
