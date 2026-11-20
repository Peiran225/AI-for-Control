#!/usr/bin/env python3
"""Decide whether the step-7 ranking loop has converged.

Usage:
    python research/tools/rank_stability.py research/projects/<slug> [--top-k 3] [--max-iter 4]

Reads ``state.json["ranking_history"]``: a list of snapshots
``{"iteration": n, "ranking": ["D2", "D1", ...], "novelty": {"D1": 4, ...}}``
appended by the orchestrator after each step-6 pass.  Prints a JSON verdict:

    {"stop": bool, "reason": str, "kendall_tau": float | null, ...}

Stop rule (from the plan): the top-k order is identical in the last two
iterations AND no candidate's novelty moved by >= 1 between them; or the
iteration cap is reached.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json  # noqa: E402


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Kendall tau-a on the items common to both rankings (None if < 2 shared)."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for x, y in combinations(common, 2):  # x precedes y in a
        if pos_b[x] < pos_b[y]:
            concordant += 1
        else:
            discordant += 1
    return round((concordant - discordant) / (concordant + discordant), 4)


def verdict(history: list[dict], top_k: int = 3, max_iter: int = 4) -> dict:
    n = len(history)
    out = {"iterations": n, "stop": False, "reason": "", "kendall_tau": None,
           "top_k_prev": None, "top_k_curr": None, "novelty_shifts": {}}
    if n == 0:
        out["reason"] = "no rankings yet"
        return out
    curr = history[-1]
    out["top_k_curr"] = curr["ranking"][:top_k]
    if n >= 2:
        prev = history[-2]
        out["top_k_prev"] = prev["ranking"][:top_k]
        out["kendall_tau"] = kendall_tau(prev["ranking"], curr["ranking"])
        shifts = {
            cid: curr["novelty"][cid] - prev["novelty"][cid]
            for cid in curr.get("novelty", {})
            if cid in prev.get("novelty", {}) and curr["novelty"][cid] != prev["novelty"][cid]
        }
        out["novelty_shifts"] = shifts
        same_top = out["top_k_prev"] == out["top_k_curr"]
        big_shift = any(abs(v) >= 1 for v in shifts.values())
        if same_top and not big_shift:
            out["stop"] = True
            out["reason"] = f"top-{top_k} unchanged across the last two iterations and no novelty shift >= 1"
            return out
        reasons = []
        if not same_top:
            reasons.append(f"top-{top_k} changed {out['top_k_prev']} -> {out['top_k_curr']}")
        if big_shift:
            reasons.append(f"novelty shifted {shifts}")
        out["reason"] = "; ".join(reasons)
    else:
        out["reason"] = "need at least two iterations"
    if n >= max_iter:
        out["stop"] = True
        out["reason"] = f"iteration cap {max_iter} reached ({out['reason']})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=4)
    args = ap.parse_args()
    history = load_json(args.project / "state.json").get("ranking_history", [])
    print(json.dumps(verdict(history, args.top_k, args.max_iter), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
