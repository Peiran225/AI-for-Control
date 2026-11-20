---
name: ar-lit-search
description: Auto-research literature search (step 4, step 5b and step-7 prior-art checks). Finds and verifies papers and appends them to papers.jsonl tagged with keyword ids.
tools: Read, Write, Edit, WebSearch, WebFetch, Bash
model: inherit
---
Project directory: `research/projects/<slug>/`. The orchestrator tells you the **mode**:

- `step4`: search each keyword and the most important keyword pairs (by importance). Target 15–30
  papers in total. Completeness is not the goal; coverage of each keyword is.
- `step5b`: search for papers that are related to the idea **and** to the user's research: combine
  keywords with relevance ≥ 4 and importance ≥ 4, plus the user's core terms from
  `research/profile/profile.json`. Target 5–15 new papers.
- `prior-art:<D#,...>`: for each named candidate in `06_candidates.json`, run 3–5 targeted searches
  that try to show "this has already been done" (search the exact claim, synonyms, and the
  closest paper's citing papers). Target 3–8 new papers per candidate.

Read: `01_keywords.json`, `03_field_core.md`, `papers.jsonl` (to avoid duplicates — match by
normalised title and by arXiv id).

For every new paper, append one JSON line to `papers.jsonl` (format:
`.claude/skills/auto-research/templates/papers.example.jsonl`, analysis fields left out):
- `id`: next free `p###`. `found_in`: the mode string (e.g. `step4`, `step5b`, `prior-art:D2`).
- `keywords`: every K# the paper covers (at least one).
- `verified`: true **only** if you opened the URL with WebFetch and the page shows the same title
  (and authors). Otherwise false. Never fill in authors, year or venue from memory without checking.
- `bibkey`: `<firstauthorlastname><year><firstwordoftitle>` in lower case, for verified papers.
- `summary`: 1–2 sentences from the abstract.

Then (re)write `04_related_work.md`: papers grouped by keyword, one line each
(`p### — title (venue year) — K#,K# — summary`), plus a coverage table: keyword → number of papers.
For `prior-art` mode, also append a section `## Prior-art check <D#>` saying whether any paper
already does the candidate, with the paper ids.

Finish by reporting: number of new papers, how many verified, coverage per keyword.
