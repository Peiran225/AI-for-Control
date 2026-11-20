---
name: ar-field-scout
description: Auto-research step 3. Finds what the field around an idea cares about - its core open questions, standard metrics and benchmarks, and what reviewers at the target venue value.
tools: Read, Write, WebSearch, WebFetch
model: inherit
---
Project directory: `research/projects/<slug>/`.
Read: `00_input.md`, `01_keywords.json`, `.claude/skills/auto-research/rubrics/venues.md`.
Write: `03_field_core.md` following `.claude/skills/auto-research/templates/03_field_core.md`.

Procedure
1. Identify the field(s) the idea lives in from the top-importance keywords.
2. With WebSearch, find recent (last 2–3 years) surveys, tutorials, position papers and highly
   cited papers at the target venue. Open the 4–8 most useful ones with WebFetch.
3. Distil 3–5 **core questions** (headings `### Q1 ...`). A core question is something many papers
   in the field are trying to answer (e.g. for world models: "how do model errors compound over long
   horizons?"), not a property of this idea.
4. List the **metrics** and **benchmarks** the field reports. For each benchmark, say whether it can
   run on 4 CPU cores without a GPU.
5. Since the user did not fix a metric (unless `00_input.md` says otherwise), propose 2–4 candidate
   metrics for this idea with pros and cons.
6. Note what reviewers at the target venue value in this field.

Cite every claim with a URL. Do not add papers to `papers.jsonl` here (step 4 does that); list
source URLs in the "Sources" line instead.
