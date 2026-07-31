# Related-work fidelity and adaptation tracks

This tree separates original-paper benchmark executions/reimplementations from adaptations to
the common tumor OCP.  The method-specific directories preserve each paper's
algorithm.  `common/` supplies provenance and comparison infrastructure only;
it never changes a method's optimizer, discretization, or training data.

The normative method contract is in `FIDELITY_CONTRACT.md`.

## Claim labels

Every normalized run uses exactly one label:

- `original-method reproduction`: an original paper benchmark; it is excluded
  from the tumor ranking even if its native objective is numerically smaller.
- `tumor adaptation`: a feasible tumor control exported as canonical `t,u` and
  eligible for common realized-`J` recomputation.
- `non-comparable adaptation`: a changed problem class that is documented but
  excluded from the tumor ranking.
- `explicitly unavailable`: a required dependency or declared runner is not
  available.  This is preferable to a silent fallback.  In particular LyZNet
  requires the real dReal binding; the common runner never substitutes a stub.

## Normalized root-manifest contract

`common.manifest.validate_root_manifest` requires, for every run:

- explicit integer RNG seeds (or an explicit deterministic/no-RNG declaration)
  and a machine-readable compute budget;
- the upstream kind and immutable revision (or an explicit reason it cannot be
  recorded), in addition to per-file source hashes;
- a selection rule fixed before training, its metric scope, confirmation that realized
  tumor `J` was unavailable to selection, and retention of every candidate;
- SHA-256 records for sources and artifacts;
- one of the claim labels above and a separate `tumor_comparable` decision;
- separate `metrics.native` and `metrics.realized` fields;
- for each completed tumor adaptation, a hashed canonical NPZ containing
  one-dimensional `t` and ZOH interval `u` arrays.

The validator rejects post-hoc selection on nominal realized `J`, conflation of
native and canonical metrics, unsafe relative paths, missing seeds/budgets, and
hash or byte-size mismatches.  A method's own manifest remains intact under its
run directory; the root manifest is an auditable normalization layer.

## Smoke orchestration

From the repository root:

```bash
python -m faithful_related_work.common.smoke \
  --output-dir /tmp/faithful_related_smokes
```

Limit the run with, for example, `--methods neural_pmp,pi_deeponet`.  The
orchestrator calls the smoke declared by each method registry entry, captures
stdout/stderr, inventories every artifact, verifies hashes, and writes
`root_manifest.json`.  It refuses to write into a non-empty output directory.

The current smoke registry covers DeepBSDE, Neural-PMP, PI-DeepONet, LyZNet,
and Adaptive HJB-NN.  The HJB-NN original smoke validates the released
satellite checkpoint on the independent author test trajectories. LyZNet first
runs its static/environment preflight; if
the real local dReal binding is missing, its root entry is explicitly
unavailable and no fake verification path is executed.

These are wiring smokes, not paper-scale experimental results.

## Normalize existing method runs

The method manifests intentionally retain their native schemas. To build one
strict root manifest from completed runs, use the explicit import adapters:

```bash
python -m faithful_related_work.common.compose \
  --artifact-root /path/containing/the/method/run/directories \
  --input deepbsde=/path/to/deepbsde/sweep_manifest.json \
  --input neural_pmp=/path/to/neural_pmp/manifest.json \
  --input pi_deeponet=/path/to/pi_deeponet/manifest.json \
  --input lyznet=/path/to/lyznet/manifest.json \
  --input hjb_nn=/path/to/hjb_nn/manifest.json \
  --output-dir /tmp/faithful_composed_manifest
```

Each adapter reads the method's actual fields and verifies every recorded hash.
If an older method manifest did not record runner-source hashes, the adapter
hashes the current audited sources but labels them explicitly as an
`import-time snapshot`; it never presents them as run-time provenance. Missing
seed, budget, selection, canonical-control, or realized/native declarations
cause a hard error rather than a guessed value. DeepBSDE and HJB-NN sweep roots
are expanded into their child runs; PI-DeepONet seed records remain separate;
the Neural-PMP selected control retains the full declared training seed set.

The completed Neural-PMP original-LQR artifact is
`paper_runs/faithful_related_work/neural_pmp_lqr_full_10seed_server`.  It contains
all ten paper-budget seeds rather than a smoke subset.  The normalized manifest
that replaces the old Neural-PMP smoke entry with this full run is
`paper_runs/faithful_related_work/final_composed_manifest_npmp_full/root_manifest.json`;
its independently recomputed tumor-only companion remains separate at
`paper_runs/faithful_related_work/final_tumor_summary_npmp_full/`.

## Canonical tumor summary

Given a normalized manifest whose paths are relative to an artifact root:

```bash
python -m faithful_related_work.common.summary \
  --manifest /path/to/root_manifest.json \
  --artifact-root /path/to/run/root \
  --source-root /path/to/repository \
  --output-dir /tmp/faithful_tumor_summary
```

The command verifies hashes, loads each completed `tumor adaptation` canonical
NPZ, and independently calls `tumor_problem.evaluate_zoh_control` with
breakpoint-aligned ZOH execution.  It checks any stored realized `J` against
the recomputation and writes `summary.json` plus `summary.csv`.  Original-paper
benchmarks, unavailable runs, failed runs, and non-comparable adaptations are
listed as excluded and never receive a tumor rank.  Native metrics remain in
their own columns and are not treated as realized objectives.

## Tests

```bash
python -m unittest tests.test_faithful_common -v
```

The tests cover schema validation, source/artifact hash verification,
claim-label eligibility, native/realized separation, no-posthoc-selection and
candidate-retention invariants, canonical objective recomputation, exclusion of
original benchmarks from tumor ranking, and no-overwrite behavior.
