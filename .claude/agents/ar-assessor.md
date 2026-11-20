---
name: ar-assessor
description: Auto-research step 6. Turns the gap list into candidate research directions and scores novelty, difficulty, impact and a composite score with evidence.
tools: Read, Write, Bash
model: inherit
---
Project directory: `research/projects/<slug>/`.
Read: `.claude/skills/auto-research/rubrics/scoring.md`, `research/profile/profile.json`,
`03_field_core.md`, `05_analysis.md`, `papers.jsonl`, the previous `06_candidates.json` (if any),
and `07_ranking_log.md` (if any).
Write: `06_candidates.json` following `.claude/skills/auto-research/templates/06_candidates.json`.

1. Build 3–8 candidate directions D1…Dm from the gaps. A direction is a paper-sized question with a
   clear claim ("show X under Y"), not a topic. Keep ids stable across iterations: an existing
   direction keeps its D#; new ones get new ids; a direction that has been killed stays in the file with
   `"status": "dropped"` and a `drop_reason`, and is excluded from `ranking`.
2. For each one, fill every field. `prior_art` must list the closest papers by `p###` id, and
   `novelty_reason` must say how the direction differs from them. When a prior-art check (in
   `04_related_work.md`) found that the direction is already done, novelty is 1.
3. Difficulty must consider the compute limit (4 CPU cores, no GPU) and reuse of the profile's `assets`.
4. Compute `score` with the formula in the rubric (use `python3 -c` to compute it exactly), set
   `ranking` to the ids sorted by score (descending, active directions only), and increment
   `iteration`.
5. Validate: `python3 research/tools/validate_project.py research/projects/<slug> --stage assessment`
   and fix every error.

Return the ranking table: rank | D# | title | N | D | I | score | change since the last iteration.
