"""Shared constants and helpers for the auto-research tools (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

STAGES = [
    "input",        # 00_input.md, state.json
    "keywords",     # 01_keywords.json, 02_background_gaps.md   (steps 1-2)
    "checkpoint_a",
    "field_core",   # 03_field_core.md                           (step 3)
    "lit_search",   # papers.jsonl, 04_related_work.md           (step 4)
    "analysis",     # papers.jsonl analysis fields, 05_analysis.md (step 5, 5b)
    "assessment",   # 06_candidates.json                          (step 6)
    "ranking_loop", # 07_ranking_log.md, state.ranking_history    (step 7)
    "checkpoint_b",
    "idea",         # 08_idea.md                                  (step 8)
    "checkpoint_c",
    "experiments",  # 09_experiments/                             (step 9)
    "paper",        # 09_paper/
    "review",       # 09_review_log.md
    "done",
]

RELATIONS = {"builds_on", "competes", "complementary", "unrelated"}

# Default composite-score weights for step 6. Difficulty enters as (6 - difficulty)
# so that easier directions score higher.
DEFAULT_WEIGHTS = {"novelty": 0.4, "impact": 0.35, "feasibility": 0.25}


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def composite_score(novelty: float, impact: float, difficulty: float, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return round(
        w["novelty"] * novelty + w["impact"] * impact + w["feasibility"] * (6 - difficulty), 3
    )
