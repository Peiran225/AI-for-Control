# Target venues: what reviewers look for

Used by `ar-field-scout` (step 3), `ar-writer` and `ar-reviewer` (step 9).
Before writing, the writer must re-check the current year's call for papers with
WebSearch (page limits and templates change every year) and note the year used in
`09_paper/README.md`.

## ML venues: NeurIPS, ICML, ICLR
- **Format** (check each year): main text about 9 pages (NeurIPS/ICLR) or 8 pages (ICML), plus unlimited references and appendix. NeurIPS requires the paper checklist. Use the official style file if it can be downloaded; otherwise use `article` with a comment saying where the official style goes.
- **What reviewers score**: soundness, significance, novelty/originality, clarity, reproducibility.
- **Typical reasons for rejection**: (1) the contribution is unclear or incremental compared with a named prior paper; (2) theory with no intuition, or experiments that do not test the claims; (3) weak baselines, only one seed, no error bars; (4) toy experiments only, with no argument for why they matter; (5) missing related work.
- **Signals of a strong paper**: one crisp question; a theorem that changes how people think, or an empirical finding with controlled ablations; honest limitations.
- For theory-leaning papers (e.g. "are these uncertainty sets equivalent?"), small tabular experiments are acceptable **if** they directly illustrate the theorem's predictions and failure cases.

## Robotics venues: CoRL, ICRA, IROS, RSS
- **Format**: CoRL about 8 pages plus references; ICRA/IROS 6 pages plus references (two-column IEEE); RSS 8 pages.
- **What reviewers score**: relevance to robotics, technical quality, experimental validation (simulation **and ideally real hardware**), comparison with robotics baselines.
- **Typical reasons for rejection**: simulation only with no sim-to-real discussion; benchmarks that are unrealistic for robots; no timing or compute analysis for control loops.
- With no GPU and no robot available here, robotics venues are high risk for a fully automatic run. The assessor should give difficulty ≥ 4 to any direction that needs hardware results.

## Simulated review form (ar-reviewer)
For each venue, score 1–10 overall and give 1–4 sub-scores for soundness,
presentation and contribution. List: summary, strengths, weaknesses (each tagged
`[major]` or `[minor]`, with the section it refers to), questions, and the
experiments or proofs that would most raise the score.
