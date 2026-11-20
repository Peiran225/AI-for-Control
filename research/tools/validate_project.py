#!/usr/bin/env python3
"""Validate an auto-research project directory.

Usage:
    python research/tools/validate_project.py research/projects/<slug> [--profile research/profile/profile.json]

Checks every artifact that exists for the stages the project has reached
(``state.json["stage"]``).  Exit code 0 when no errors, 1 otherwise.  The
report is printed as JSON so that the orchestrator can feed it back to the
sub-agent that produced the faulty file.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_WEIGHTS,
    RELATIONS,
    STAGES,
    composite_score,
    load_json,
    load_jsonl,
    stage_index,
)

STATE_KEYS = {"slug", "idea", "venue", "stage", "completed", "checkpoints", "ranking_history"}
PAPER_KEYS = {"id", "title", "authors", "year", "url", "keywords", "verified", "found_in"}
CANDIDATE_KEYS = {
    "id", "title", "description", "targets", "prior_art",
    "novelty", "novelty_reason", "difficulty", "difficulty_reason",
    "impact", "impact_reason", "score",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")

    def as_dict(self) -> dict:
        return {"ok": not self.errors, "errors": self.errors, "warnings": self.warnings}


def _score_ok(value, where: str, field: str, rep: Report) -> bool:
    if not isinstance(value, int) or not 1 <= value <= 5:
        rep.err(where, f"{field} must be an integer in 1..5, got {value!r}")
        return False
    return True


# ----------------------------------------------------------------- stage checks
def check_state(proj: Path, rep: Report) -> dict | None:
    path = proj / "state.json"
    if not path.exists():
        rep.err("state.json", "missing")
        return None
    state = load_json(path)
    missing = STATE_KEYS - state.keys()
    if missing:
        rep.err("state.json", f"missing keys {sorted(missing)}")
    if state.get("stage") not in STAGES:
        rep.err("state.json", f"unknown stage {state.get('stage')!r}")
        return None
    if not (proj / "00_input.md").exists():
        rep.err("00_input.md", "missing")
    return state


def check_keywords(proj: Path, rep: Report, profile_ids: set[str] | None) -> dict[str, dict]:
    path = proj / "01_keywords.json"
    if not path.exists():
        rep.err("01_keywords.json", "missing")
        return {}
    data = load_json(path)
    kws = data.get("keywords", [])
    if not 3 <= len(kws) <= 8:
        rep.err("01_keywords.json", f"expected 3-8 keywords, got {len(kws)}")
    out: dict[str, dict] = {}
    for kw in kws:
        kid = kw.get("id", "?")
        where = f"01_keywords.json[{kid}]"
        if not re.fullmatch(r"K\d+", str(kid)):
            rep.err(where, "id must look like K1, K2, ...")
        if kid in out:
            rep.err(where, "duplicate id")
        out[kid] = kw
        for field in ("term", "relevance_reason", "importance_reason"):
            if not str(kw.get(field, "")).strip():
                rep.err(where, f"empty {field}")
        rel_ok = _score_ok(kw.get("relevance"), where, "relevance", rep)
        imp_ok = _score_ok(kw.get("importance"), where, "importance", rep)
        evidence = kw.get("relevance_evidence", [])
        if kw.get("relevance", 0) >= 3 and not evidence:
            rep.err(where, "relevance >= 3 requires relevance_evidence (profile paper ids)")
        if profile_ids is not None:
            unknown = [e for e in evidence if e not in profile_ids]
            if unknown:
                rep.err(where, f"relevance_evidence ids not in profile.json: {unknown}")
        if rel_ok and imp_ok:
            expected = kw["relevance"] <= 2 and kw["importance"] >= 4
            if bool(kw.get("needs_background")) != expected:
                rep.err(where, f"needs_background should be {expected} (relevance<=2 and importance>=4)")
    order = [k.get("relevance", 0) for k in kws]
    if order != sorted(order, reverse=True):
        rep.err("01_keywords.json", "keywords must be sorted by relevance (descending)")

    gaps = [k for k in kws if k.get("needs_background")]
    gap_path = proj / "02_background_gaps.md"
    if gaps:
        if not gap_path.exists():
            rep.err("02_background_gaps.md", "missing although some keywords need background")
        else:
            text = gap_path.read_text(encoding="utf-8")
            headings = "\n".join(l for l in text.splitlines() if l.startswith("#"))
            for k in gaps:
                if k["id"] not in headings:
                    rep.err("02_background_gaps.md", f"no heading for {k['id']} ({k.get('term')})")
    return out


def check_field_core(proj: Path, rep: Report) -> set[str]:
    path = proj / "03_field_core.md"
    if not path.exists():
        rep.err("03_field_core.md", "missing")
        return set()
    text = path.read_text(encoding="utf-8")
    qids = set(re.findall(r"^#+\s*(Q\d+)\b", text, flags=re.M))
    if len(qids) < 3:
        rep.err("03_field_core.md", f"need >= 3 core questions as headings '### Q1 ...', found {sorted(qids)}")
    for section in ("Metrics", "Benchmarks"):
        if not re.search(rf"^#+\s*{section}", text, flags=re.M | re.I):
            rep.err("03_field_core.md", f"missing '## {section}' section")
    return qids


def check_papers(proj: Path, rep: Report, kw_ids: set[str], qids: set[str], analysed: bool) -> dict[str, dict]:
    path = proj / "papers.jsonl"
    if not path.exists():
        rep.err("papers.jsonl", "missing")
        return {}
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        rep.err("papers.jsonl", str(exc))
        return {}
    papers: dict[str, dict] = {}
    bibkeys: set[str] = set()
    for p in rows:
        pid = p.get("id", "?")
        where = f"papers.jsonl[{pid}]"
        missing = PAPER_KEYS - p.keys()
        if missing:
            rep.err(where, f"missing keys {sorted(missing)}")
        if pid in papers:
            rep.err(where, "duplicate id")
        papers[pid] = p
        if not str(p.get("url", "")).startswith("http"):
            rep.err(where, "url must start with http")
        if not isinstance(p.get("verified"), bool):
            rep.err(where, "verified must be a boolean")
        bad_kw = [k for k in p.get("keywords", []) if k not in kw_ids]
        if bad_kw:
            rep.err(where, f"unknown keyword ids {bad_kw}")
        if not p.get("keywords"):
            rep.err(where, "must be tagged with at least one keyword")
        if p.get("verified") and not p.get("bibkey"):
            rep.err(where, "verified papers need a bibkey")
        if p.get("bibkey"):
            if p["bibkey"] in bibkeys:
                rep.err(where, f"duplicate bibkey {p['bibkey']}")
            bibkeys.add(p["bibkey"])
        if analysed:
            if p.get("relation") not in RELATIONS:
                rep.err(where, f"relation must be one of {sorted(RELATIONS)}")
            for field in ("not_done", "matters"):
                if not str(p.get(field, "")).strip():
                    rep.err(where, f"empty analysis field {field}")
            bad_q = [q for q in p.get("serves_questions", []) if q not in qids]
            if bad_q:
                rep.err(where, f"serves_questions refers to unknown {bad_q}")
    if not (proj / "04_related_work.md").exists():
        rep.err("04_related_work.md", "missing")
    if analysed and not (proj / "05_analysis.md").exists():
        rep.err("05_analysis.md", "missing")
    return papers


def check_candidates(proj: Path, rep: Report, papers: dict[str, dict], qids: set[str]) -> None:
    path = proj / "06_candidates.json"
    if not path.exists():
        rep.err("06_candidates.json", "missing")
        return
    data = load_json(path)
    weights = data.get("weights", DEFAULT_WEIGHTS)
    if set(weights) != set(DEFAULT_WEIGHTS):
        rep.err("06_candidates.json", f"weights keys must be {sorted(DEFAULT_WEIGHTS)}")
        weights = DEFAULT_WEIGHTS
    cands = data.get("candidates", [])
    for c in cands:
        if c.get("status", "active") == "dropped" and not str(c.get("drop_reason", "")).strip():
            rep.err(f"06_candidates.json[{c.get('id', '?')}]", "dropped candidates need a drop_reason")
    if len([c for c in cands if c.get("status", "active") != "dropped"]) < 2:
        rep.err("06_candidates.json", "need at least 2 candidate directions")
    seen = set()
    for c in cands:
        cid = c.get("id", "?")
        where = f"06_candidates.json[{cid}]"
        missing = CANDIDATE_KEYS - c.keys()
        if missing:
            rep.err(where, f"missing keys {sorted(missing)}")
            continue
        if cid in seen:
            rep.err(where, "duplicate id")
        seen.add(cid)
        ok = all(_score_ok(c.get(f), where, f, rep) for f in ("novelty", "difficulty", "impact"))
        if not c["prior_art"]:
            rep.err(where, "prior_art must name the closest existing papers")
        unknown = [p for p in c["prior_art"] if p not in papers]
        if unknown:
            rep.err(where, f"prior_art ids not in papers.jsonl: {unknown}")
        bad_q = [q for q in c["targets"] if q not in qids]
        if bad_q or not c["targets"]:
            rep.err(where, f"targets must reference core questions, bad={bad_q}")
        if ok:
            expected = composite_score(c["novelty"], c["impact"], c["difficulty"], weights)
            if abs(float(c["score"]) - expected) > 0.01:
                rep.err(where, f"score {c['score']} != formula value {expected}")
    ranking = data.get("ranking")
    if ranking is not None:
        active = [c for c in cands if c.get("status", "active") != "dropped"]
        by_score = [c["id"] for c in sorted(active, key=lambda c: -float(c.get("score", 0)))]
        if [r for r in ranking] != by_score:
            rep.err("06_candidates.json", f"ranking {ranking} does not follow scores {by_score}")


def check_idea(proj: Path, rep: Report) -> None:
    path = proj / "08_idea.md"
    if not path.exists():
        rep.err("08_idea.md", "missing")
        return
    text = path.read_text(encoding="utf-8")
    for section in ("Problem", "Claims", "Method", "Evidence plan", "Baselines", "Metrics", "Compute budget", "Risks"):
        if not re.search(rf"^#+\s*{section}", text, flags=re.M | re.I):
            rep.err("08_idea.md", f"missing '## {section}' section")
    if not re.search(r"\bC1\b", text):
        rep.err("08_idea.md", "claims must be numbered C1, C2, ...")


def check_experiments(proj: Path, rep: Report) -> None:
    res = proj / "09_experiments" / "results"
    csvs = sorted(res.glob("*.csv")) if res.exists() else []
    if not csvs:
        rep.err("09_experiments/results", "no CSV results")
        return
    for path in csvs:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            rep.err(path.name, "empty CSV")
            continue
        if "seed" in rows[0]:
            seeds = {r["seed"] for r in rows}
            if len(seeds) < 3:
                rep.err(path.name, f"need >= 3 seeds, found {sorted(seeds)}")
    if not (proj / "09_experiments" / "requirements.txt").exists():
        rep.warn("09_experiments", "requirements.txt missing")


_CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
_BIB_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
# Numbers that look like reported results: decimals or percentages.
_NUM_RE = re.compile(r"(?<![\\\w{])(\d+\.\d+|\d+(?:\.\d+)?\s*\\%)")
_SAFE_CMD_RE = re.compile(
    r"\\(?:includegraphics|vspace|hspace|setlength|resizebox|scalebox|usepackage|documentclass|"
    r"label|ref|cref|eqref|input|begin|end|url|href|addtolength|linespread)\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*"
)
_LENGTH_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:pt|em|ex|cm|mm|in|\\(?:linewidth|textwidth|columnwidth))")


def strip_tex(text: str) -> list[tuple[int, str]]:
    """Return (lineno, line) pairs with comments, lengths and layout commands removed.

    Lines carrying a trailing ``% num-ok`` marker are dropped entirely: they are
    reserved for constants that are part of the problem statement (e.g. a
    discount factor in a theorem), which the reviewer agent checks by hand.
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r"(?<!\\)%\s*num-ok", line):
            continue
        line = re.sub(r"(?<!\\)%.*$", "", line)
        line = _SAFE_CMD_RE.sub("", line)
        line = _LENGTH_RE.sub("", line)
        out.append((i, line))
    return out


def check_paper(proj: Path, rep: Report, papers: dict[str, dict]) -> None:
    pdir = proj / "09_paper"
    main = pdir / "main.tex"
    if not main.exists():
        rep.err("09_paper/main.tex", "missing")
        return
    bib = pdir / "refs.bib"
    bib_keys = set(_BIB_RE.findall(bib.read_text(encoding="utf-8"))) if bib.exists() else set()
    verified = {p.get("bibkey") for p in papers.values() if p.get("verified")}
    for key in sorted(bib_keys - verified):
        rep.err("09_paper/refs.bib", f"entry {key!r} is not a verified paper in papers.jsonl")
    tex_files = [main] + sorted((pdir / "sections").glob("*.tex")) if (pdir / "sections").exists() else [main]
    for tex in tex_files:
        text = tex.read_text(encoding="utf-8")
        rel = tex.relative_to(proj)
        for match in _CITE_RE.finditer(text):
            for key in (k.strip() for k in match.group(1).split(",")):
                if key and key not in bib_keys:
                    rep.err(str(rel), f"\\cite{{{key}}} not in refs.bib")
        for lineno, line in strip_tex(text):
            for num in _NUM_RE.findall(line):
                rep.err(f"{rel}:{lineno}", f"literal result-like number {num.strip()!r}; use a macro from results_macros.tex")
    if not (pdir / "results_macros.tex").exists():
        rep.err("09_paper/results_macros.tex", "missing; run make_results_macros.py")


# ----------------------------------------------------------------------- driver
def validate(proj: Path, profile: Path | None = None, completed_stage: str | None = None) -> Report:
    """Validate ``proj``.  ``completed_stage`` checks artifacts up to and including
    that stage regardless of ``state.json["stage"]`` (used right after a sub-agent
    finishes, before the orchestrator advances the state)."""
    rep = Report()
    state = check_state(proj, rep)
    if state is None:
        return rep
    reached = stage_index(state["stage"])
    if completed_stage is not None:
        reached = max(reached, stage_index(completed_stage) + 1)

    def done(stage: str) -> bool:
        # A stage's artifacts must exist once the project has moved past it.
        return reached > stage_index(stage)

    profile_ids = None
    if profile and profile.exists():
        profile_ids = {p["id"] for p in load_json(profile).get("papers", [])}

    kws: dict[str, dict] = {}
    qids: set[str] = set()
    papers: dict[str, dict] = {}
    if done("keywords"):
        kws = check_keywords(proj, rep, profile_ids)
    if done("field_core"):
        qids = check_field_core(proj, rep)
    if done("lit_search"):
        papers = check_papers(proj, rep, set(kws), qids, analysed=done("analysis"))
    if done("assessment"):
        check_candidates(proj, rep, papers, qids)
    if done("ranking_loop"):
        if not (proj / "07_ranking_log.md").exists():
            rep.err("07_ranking_log.md", "missing")
        if len(state.get("ranking_history", [])) < 2:
            rep.err("state.json", "ranking_history needs >= 2 iterations before leaving the ranking loop")
    if done("idea"):
        check_idea(proj, rep)
    if done("experiments"):
        check_experiments(proj, rep)
    if done("paper"):
        check_paper(proj, rep, papers)
    if done("review") and not (proj / "09_review_log.md").exists():
        rep.err("09_review_log.md", "missing")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    ap.add_argument("--profile", type=Path, default=Path("research/profile/profile.json"))
    ap.add_argument("--stage", choices=STAGES, help="validate artifacts up to and including this completed stage")
    args = ap.parse_args()
    rep = validate(args.project, args.profile, args.stage)
    print(json.dumps(rep.as_dict(), indent=2, ensure_ascii=False))
    return 0 if not rep.errors else 1


if __name__ == "__main__":
    sys.exit(main())
