# Scoring rubric (shared by all auto-research agents)

All scores are integers 1–5. Every score must come with a one-sentence reason that
names concrete evidence (a profile paper id `P#`, a paper id `p###`, or a core
question `Q#`). A score with no evidence is invalid.

## Keyword relevance to my background (step 2)
| Score | Anchor |
|---|---|
| 5 | A central object or method of at least one of my papers (I have published on exactly this). |
| 4 | I used it as a key tool or baseline in a paper, but it was not the contribution. |
| 3 | Adjacent: shares the math or the setting with my work (e.g. same optimality conditions, same dynamics class). |
| 2 | I know the basics; I have never used it in a paper. |
| 1 | New to me. |

`relevance_evidence` lists the profile paper ids (`P1`, `P2`, ...) that justify scores ≥ 3.

## Keyword importance to the idea (step 2)
| Score | Anchor |
|---|---|
| 5 | Remove it and the idea no longer exists. |
| 4 | Required for the main claim or the main experiment. |
| 3 | Needed for one section (a baseline family, a secondary experiment). |
| 2 | Context only (related-work paragraph). |
| 1 | Tangential. |

A keyword with **relevance ≤ 2 and importance ≥ 4** gets `needs_background: true`
and a background note in `02_background_gaps.md`.

## Novelty of a candidate direction (step 6)
Always compare against the closest paper(s) in `prior_art`.
| Score | Anchor |
|---|---|
| 5 | No paper found that asks this question; the closest work studies a clearly different object. |
| 4 | Closest work asks a related question in a different setting (e.g. tabular vs. continuous, finite vs. infinite horizon), and the transfer is non-trivial. |
| 3 | Closest work covers a special case; ours is a strict generalisation with a new insight. |
| 2 | Incremental: new combination of known pieces, or a new benchmark for a known method. |
| 1 | Already done (a paper in `prior_art` essentially contains it). |

## Difficulty (step 6) — higher means harder
Consider three factors and take the max, then adjust by at most 1:
- **Theory**: 1 = definitions + known lemmas; 3 = a new theorem with a standard proof technique; 5 = an open problem.
- **Compute**: the container has 4 CPU cores, 15 GB RAM and no GPU. 1 = minutes; 3 = a few CPU-hours per figure; 5 = needs GPUs or large simulators (MuJoCo-scale RL with many seeds).
- **Engineering / reuse**: 1 = reuse code listed under `assets` in `profile.json`; 5 = build a new simulator or dataset.

## Impact (step 6)
| Score | Anchor |
|---|---|
| 5 | Directly answers a core question `Q#` from `03_field_core.md` and would change how people work on it. |
| 4 | Clear progress on a core question; reviewers at the target venue would care. |
| 3 | Useful insight for a sub-community. |
| 2 | Mostly of interest to me. |
| 1 | Does not touch any core question. |

## Composite score
`score = 0.40·novelty + 0.35·impact + 0.25·(6 − difficulty)`, rounded to 3 decimals.
The weights live in `06_candidates.json["weights"]` and may be changed at checkpoint B;
`research/tools/validate_project.py` recomputes the score and rejects mismatches.
