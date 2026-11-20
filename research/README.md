# Auto research agent

A Claude Code–native pipeline that turns a one-sentence idea into ranked research
directions, a chosen idea, experiments and a LaTeX paper.

## Usage (inside Claude Code, from the repository root)
1. Fill in `research/profile/my_papers.md` with your papers.
2. `/auto-research profile` builds `research/profile/profile.json`.
3. `/auto-research new "Are different forms of uncertainty in MDPs equivalent?" --venue neurips`
4. Answer the three checkpoints (after keywords, after the ranking loop, after the idea).
5. `/auto-research resume <slug>` continues an interrupted run; `/auto-research status <slug>` shows progress.

## Pieces
| Path | Role |
|---|---|
| `.claude/skills/auto-research/SKILL.md` | Orchestrator: stage machine, checkpoints, ranking loop, review loop |
| `.claude/skills/auto-research/rubrics/` | 1–5 scoring anchors and venue expectations |
| `.claude/skills/auto-research/templates/` | Formats of every artifact |
| `.claude/agents/ar-*.md` | Sub-agents: profiler, keywords, field-scout, lit-search, paper-analyst, assessor, experimenter, writer, reviewer |
| `research/tools/validate_project.py` | Checks each stage's files, citation integrity (only verified papers) and number provenance (only macros) |
| `research/tools/rank_stability.py` | Stop rule for the ranking loop (top-3 stable, no novelty shift ≥ 1, at most 4 iterations) |
| `research/tools/make_results_macros.py` | `results/summary*.csv` → `09_paper/results_macros.tex` |
| `research/projects/<slug>/` | One directory per idea; see the layout in `SKILL.md` |

Tests: `python -m pytest -q tests/test_research_tools.py`.

## Pipeline → files
| Step | Output |
|---|---|
| 1–2 keywords + relevance | `01_keywords.json`, `02_background_gaps.md` |
| 3 field core questions | `03_field_core.md` |
| 4 related work | `papers.jsonl`, `04_related_work.md` |
| 5 per-paper analysis (+5b) | analysis fields in `papers.jsonl`, `05_analysis.md` |
| 6 novelty / difficulty | `06_candidates.json` |
| 7 ranking loop | `state.json.ranking_history`, `07_ranking_log.md` |
| 8 first idea | `08_idea.md` |
| 9 paper | `09_experiments/`, `09_paper/`, `09_review_log.md` |
