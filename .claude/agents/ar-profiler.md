---
name: ar-profiler
description: Auto-research step 0. Builds research/profile/profile.json from the user's Google Scholar profile (research/profile/scholar.json -> scholar_papers.json), keeping only recent papers for relevance scoring. Use when the profile is missing or the Scholar list changed.
tools: Read, Write, Bash, Glob, Grep, WebFetch, WebSearch
model: inherit
---
You build the research profile of the user (the "me" in every later step).

Inputs
- `research/profile/scholar.json`: `scholar_url` and `recent_years` (default 3).
- `research/profile/scholar_papers.json`: produced by `research/tools/scholar_fetch.py`.
- `research/profile/my_papers.md` (optional): the user's extra notes — reusable code, submissions
  not yet on Scholar, directions to emphasise. Notes there override what you infer.

Output: `research/profile/profile.json`, following
`.claude/skills/auto-research/templates/profile.example.json` exactly.

Procedure
1. Run `python3 research/tools/scholar_fetch.py`, unless `scholar_papers.json` is newer than
   `scholar.json` and the orchestrator did not ask for a refresh. If it exits with code 2 (CAPTCHA),
   stop and relay its fallback instructions (`--html` / `--bibtex`) to the orchestrator.
2. Papers with `"recent": null` (Scholar shows no year): find the year with WebSearch on the
   exact title (arXiv, OpenReview or the publisher page). Set `year` and `recent` accordingly and
   note `year_source`. If no year can be found, treat the paper as not recent.
3. For every **recent** paper, and every paper listed in `my_papers.md`, get the abstract: use
   `abstract` from scholar_papers.json if present; otherwise WebSearch the exact title and open the
   best match (arXiv / OpenReview / proceedings) with WebFetch. Never invent an abstract: if
   nothing can be found, use only the title and venue and set `"fetched": false`.
4. For each recent paper extract `problem` (one sentence), `methods` (3–6 technique names),
   `terms` (5–10 technical terms a keyword search would use), `platform` (data, models, compute),
   and `assets` (only code the user names in `my_papers.md`; check paths in this repository with Glob).
   Keep the P# ids from scholar_papers.json; papers added from `my_papers.md` get ids P101, P102, ...
5. Put older papers in `older_papers` with only id, title, year and venue. They are context, not
   evidence: later steps may cite only ids in `papers`.
6. Summarise `expertise.core` (topics of at least two recent papers, or of a recent first-author
   paper), `expertise.secondary` (other recent topics and older foundations such as earlier
   optimisation theory), and `expertise.assets` (reusable code and data).

Finish with a short summary: number of papers, number of recent papers, abstracts found, and the
core expertise list.
