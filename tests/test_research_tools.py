"""Tests for the deterministic auto-research tools under research/tools/."""
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "research" / "tools"
sys.path.insert(0, str(TOOLS))

from common import composite_score  # noqa: E402
from make_results_macros import build_macros, macro_name  # noqa: E402
from rank_stability import kendall_tau, verdict  # noqa: E402
from validate_project import validate  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _paper(pid, bibkey, verified=True, **extra):
    row = {
        "id": pid, "title": f"Paper {pid}", "authors": ["A. Author"], "year": 2025,
        "venue": "NeurIPS", "url": f"https://arxiv.org/abs/2501.0000{pid[-1]}",
        "keywords": ["K1"], "verified": verified, "found_in": "step4", "bibkey": bibkey,
        "relation": "competes", "not_done": "no tabular analysis", "matters": "answers Q1",
        "serves_questions": ["Q1"],
    }
    row.update(extra)
    return row


def _candidate(cid, novelty, impact, difficulty, prior_art=("p1",)):
    return {
        "id": cid, "title": cid, "description": "d", "targets": ["Q1"],
        "prior_art": list(prior_art), "novelty": novelty, "novelty_reason": "r",
        "difficulty": difficulty, "difficulty_reason": "r", "impact": impact,
        "impact_reason": "r", "score": composite_score(novelty, impact, difficulty),
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    state = {
        "slug": "proj", "idea": "x", "venue": "neurips", "stage": "done", "completed": [],
        "checkpoints": {}, "ranking_history": [
            {"iteration": 1, "ranking": ["D1", "D2"], "novelty": {"D1": 4, "D2": 3}},
            {"iteration": 2, "ranking": ["D1", "D2"], "novelty": {"D1": 4, "D2": 3}},
        ],
    }
    _write(proj / "state.json", json.dumps(state))
    _write(proj / "00_input.md", "# Input\n")
    keywords = {"keywords": [
        {"id": "K1", "term": "robust MDP", "relevance": 4, "relevance_evidence": ["P1"],
         "relevance_reason": "r", "importance": 5, "importance_reason": "r", "needs_background": False},
        {"id": "K2", "term": "bayesian RL", "relevance": 2, "relevance_evidence": [],
         "relevance_reason": "r", "importance": 4, "importance_reason": "r", "needs_background": True},
        {"id": "K3", "term": "risk measures", "relevance": 1, "relevance_evidence": [],
         "relevance_reason": "r", "importance": 2, "importance_reason": "r", "needs_background": False},
    ]}
    _write(proj / "01_keywords.json", json.dumps(keywords))
    _write(proj / "02_background_gaps.md", "# Gaps\n## K2 bayesian RL\ntext\n")
    _write(proj / "03_field_core.md",
           "# Core\n### Q1 a\n### Q2 b\n### Q3 c\n## Metrics\n- regret\n## Benchmarks\n- gridworld\n")
    papers = [_paper("p1", "smith2025"), _paper("p2", "lee2024")]
    _write(proj / "papers.jsonl", "\n".join(json.dumps(p) for p in papers) + "\n")
    _write(proj / "04_related_work.md", "rw")
    _write(proj / "05_analysis.md", "an")
    cands = [_candidate("D1", 4, 4, 2), _candidate("D2", 3, 3, 3, ("p2",))]
    _write(proj / "06_candidates.json", json.dumps({"candidates": cands, "ranking": ["D1", "D2"]}))
    _write(proj / "07_ranking_log.md", "log")
    _write(proj / "08_idea.md", "\n".join(f"## {s}\nC1 x" for s in (
        "Problem", "Claims", "Method", "Evidence plan", "Baselines", "Metrics", "Compute budget", "Risks")))
    _write(proj / "09_experiments" / "requirements.txt", "numpy\n")
    _write(proj / "09_experiments" / "results" / "runs.csv", "seed,ret\n0,1\n1,2\n2,3\n")
    _write(proj / "09_paper" / "results_macros.tex", "\\newcommand{\\rGap}{0.12}\n")
    _write(proj / "09_paper" / "refs.bib", "@article{smith2025,\n title={x}}\n@inproceedings{lee2024, title={y}}\n")
    _write(proj / "09_paper" / "main.tex",
           "\\documentclass{article}\n\\input{results_macros}\n"
           "\\includegraphics[width=0.8\\linewidth]{f.pdf}\n"
           "As shown by \\citet{smith2025}, the gap is \\rGap{} \\citep{lee2024}.\n"
           "We use $\\gamma = 0.99$. % num-ok\n")
    _write(proj / "09_review_log.md", "review")
    return proj


def test_valid_project_passes(project):
    rep = validate(project)
    assert rep.errors == []


def test_unverified_citation_rejected(project):
    rows = [json.loads(l) for l in (project / "papers.jsonl").read_text().splitlines()]
    rows[1]["verified"] = False
    (project / "papers.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    errors = validate(project).errors
    assert any("lee2024" in e and "not a verified paper" in e for e in errors)


def test_cite_missing_from_bib(project):
    main = project / "09_paper" / "main.tex"
    main.write_text(main.read_text() + "See \\cite{ghost2023}.\n")
    assert any("ghost2023" in e for e in validate(project).errors)


def test_literal_number_rejected(project):
    main = project / "09_paper" / "main.tex"
    main.write_text(main.read_text() + "Our method improves return by 12.5\\% over PPO.\n")
    errors = validate(project).errors
    assert any("literal result-like number" in e for e in errors)


def test_needs_background_consistency(project):
    data = json.loads((project / "01_keywords.json").read_text())
    data["keywords"][1]["needs_background"] = False
    (project / "01_keywords.json").write_text(json.dumps(data))
    assert any("needs_background should be True" in e for e in validate(project).errors)


def test_background_heading_required(project):
    (project / "02_background_gaps.md").write_text("# Gaps\nnothing\n")
    assert any("no heading for K2" in e for e in validate(project).errors)


def test_profile_evidence_checked(project, tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"papers": [{"id": "P9"}]}))
    assert any("not in profile.json" in e for e in validate(project, profile).errors)


def test_candidate_score_and_prior_art(project):
    data = json.loads((project / "06_candidates.json").read_text())
    data["candidates"][0]["score"] = 9.9
    data["candidates"][1]["prior_art"] = ["p404"]
    (project / "06_candidates.json").write_text(json.dumps(data))
    errors = validate(project).errors
    assert any("formula value" in e for e in errors)
    assert any("p404" in e for e in errors)


def test_too_few_seeds(project):
    (project / "09_experiments" / "results" / "runs.csv").write_text("seed,ret\n0,1\n1,2\n")
    assert any(">= 3 seeds" in e for e in validate(project).errors)


def test_early_stage_skips_later_artifacts(project):
    state = json.loads((project / "state.json").read_text())
    state["stage"] = "field_core"  # only keywords are complete
    (project / "state.json").write_text(json.dumps(state))
    (project / "09_paper" / "main.tex").unlink()
    (project / "03_field_core.md").unlink()
    assert validate(project).errors == []
    assert any("03_field_core.md" in e for e in validate(project, completed_stage="field_core").errors)


def test_kendall_tau():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a"], ["a"]) is None


def test_verdict_rules():
    h1 = {"iteration": 1, "ranking": ["D1", "D2", "D3", "D4"], "novelty": {"D1": 4, "D2": 3, "D3": 3}}
    h2_same = {"iteration": 2, "ranking": ["D1", "D2", "D3", "D5"], "novelty": {"D1": 4, "D2": 3, "D3": 3}}
    h2_shift = {"iteration": 2, "ranking": ["D1", "D2", "D3"], "novelty": {"D1": 2, "D2": 3, "D3": 3}}
    h2_swap = {"iteration": 2, "ranking": ["D2", "D1", "D3"], "novelty": {"D1": 4, "D2": 3, "D3": 3}}
    assert not verdict([h1])["stop"]
    assert verdict([h1, h2_same])["stop"]
    assert not verdict([h1, h2_shift])["stop"]
    assert not verdict([h1, h2_swap])["stop"]
    capped = verdict([h1, h2_swap, h1, h2_swap], max_iter=4)
    assert capped["stop"] and "cap" in capped["reason"]


def test_macro_generation(tmp_path):
    assert macro_name("robust_vi_gap_gamma09") == "rRobustViGapGammaZeroNine"
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("key,mean,std,n,decimals\nvi_gap,0.12345,0.01,5,3\nreturn,10,,,\n")
    tex = build_macros([csv_path])
    assert "\\newcommand{\\rViGap}{0.123}" in tex
    assert "\\newcommand{\\rViGapPM}{0.123\\ensuremath{\\pm}0.010}" in tex
    assert "\\newcommand{\\rViGapN}{5}" in tex
    assert "\\newcommand{\\rReturn}{10.00}" in tex
    with pytest.raises(ValueError):
        macro_name("bad-key")


def test_dropped_candidate_excluded_from_ranking(project):
    data = json.loads((project / "06_candidates.json").read_text())
    data["candidates"].append(dict(_candidate("D3", 5, 5, 1), status="dropped", drop_reason="done by p2"))
    (project / "06_candidates.json").write_text(json.dumps(data))
    assert validate(project).errors == []
    del data["candidates"][-1]["drop_reason"]
    (project / "06_candidates.json").write_text(json.dumps(data))
    assert any("drop_reason" in e for e in validate(project).errors)


# ------------------------------------------------------------- scholar_fetch
import datetime as _dt  # noqa: E402

from scholar_fetch import (  # noqa: E402
    finalise, is_blocked, parse_arxiv_feed, parse_bibtex, parse_profile_html, user_id_from_url,
)

_ROW = ('<tr class="gsc_a_tr"><td class="gsc_a_t"><a href="/citations?view_op=view_citation&amp;hl=en'
        '&amp;user=U1&amp;citation_for_view=U1:abc" class="gsc_a_at">{title}</a>'
        '<div class="gs_gray">A Author, B Author</div><div class="gs_gray">{venue}'
        '<span class="gs_oph">, {year}</span></div></td><td class="gsc_a_c">'
        '<a href="" class="gsc_a_ac gs_ibl">{cites}</a></td><td class="gsc_a_y">'
        '<span class="gsc_a_h gsc_a_hc gs_ibl">{year}</span></td></tr>')


def _profile_page(rows, more_disabled=True):
    btn = '<button id="gsc_bpf_more" {} class="gs_btnPD">'.format("disabled" if more_disabled else "")
    return f'<div id="gsc_prf_in">Jane Doe</div><table>{"".join(rows)}</table>{btn}'


def test_parse_scholar_profile():
    page = _profile_page([
        _ROW.format(title="Bilevel Things &amp; More", venue="arXiv preprint arXiv:2610.05290", year="2026", cites="3"),
        _ROW.format(title="Old Paper", venue="ICML", year="2019", cites=""),
        _ROW.format(title="No Year", venue="", year="", cites=""),
    ])
    papers, name, more = parse_profile_html(page)
    assert name == "Jane Doe" and not more
    assert [p["title"] for p in papers] == ["Bilevel Things & More", "Old Paper", "No Year"]
    assert papers[0]["authors"] == ["A Author", "B Author"]
    assert papers[0]["venue"] == "arXiv preprint arXiv:2610.05290"
    assert papers[0]["year"] == 2026 and papers[0]["citations"] == 3
    assert papers[0]["scholar_url"].startswith("https://scholar.google.com/citations?view_op=view_citation&hl=en&user=U1")
    assert papers[1]["citations"] == 0 and papers[2]["year"] is None
    assert parse_profile_html(_profile_page([], more_disabled=False))[2] is True


def test_recent_flag_and_ids():
    papers = finalise(
        [{"title": "a", "year": 2022}, {"title": "b", "year": None}, {"title": "c", "year": 2026},
         {"title": "d", "year": 2024}],
        recent_years=3, today=_dt.date(2026, 11, 20))
    assert [(p["id"], p["title"], p["recent"]) for p in papers] == [
        ("P1", "c", True), ("P2", "d", True), ("P3", "a", False), ("P4", "b", None)]


def test_captcha_detection_and_user_id():
    assert is_blocked('<div id="gs_captcha_ccl">')
    assert is_blocked("Our systems have detected unusual traffic from your computer")
    assert not is_blocked(_profile_page([]))
    assert user_id_from_url("https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=SXJ4R24AAAAJ") == "SXJ4R24AAAAJ"
    with pytest.raises(ValueError):
        user_id_from_url("https://scholar.google.com/")


def test_parse_bibtex_export():
    bib = ("@article{yu2025fast,\n  title={A Fast {Federated} Method},\n  author={Yu, Peiran and Huang, Heng},\n"
           "  journal={IEEE TNNLS},\n  year={2025}\n}\n\n@inproceedings{x,\n  title={Second},\n"
           "  booktitle={NeurIPS},\n  year={2023},\n  author={A, B}\n}\n")
    papers = parse_bibtex(bib)
    assert [p["title"] for p in papers] == ["A Fast Federated Method", "Second"]
    assert papers[0]["authors"] == ["Yu, Peiran", "Huang, Heng"] and papers[0]["year"] == 2025
    assert papers[1]["venue"] == "NeurIPS"


def test_parse_arxiv_feed():
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2610.05290v1</id>'
           '<published>2026-10-06T00:00:00Z</published><title>Smoothed  Gradient\n Method</title>'
           '<summary> We study\n bilevel problems. </summary></entry></feed>')
    [e] = parse_arxiv_feed(xml)
    assert e == {"title": "Smoothed Gradient Method", "abstract": "We study bilevel problems.",
                 "url": "https://arxiv.org/abs/2610.05290v1", "year": 2026}
