#!/usr/bin/env python3
"""Fetch the user's publication list from Google Scholar (stdlib only).

Usage:
    python research/tools/scholar_fetch.py                       # uses research/profile/scholar.json
    python research/tools/scholar_fetch.py --html saved.html     # parse a profile page saved from a browser
    python research/tools/scholar_fetch.py --bibtex scholar.bib  # parse Scholar's "Export -> BibTeX"

Writes ``research/profile/scholar_papers.json``.  Each paper gets an id P1, P2,
... (newest first), the Scholar metadata, an arXiv link and abstract when one
can be matched, and ``recent`` = published within the last ``recent_years``
years (``null`` when no year is known).

Google Scholar sometimes answers automated requests with a CAPTCHA.  The
script then stops with exit code 2 and explains the two offline fallbacks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

SCHOLAR = "https://scholar.google.com"
ARXIV_API = "https://export.arxiv.org/api/query"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
PAGE_SIZE = 100
ATOM = {"a": "http://www.w3.org/2005/Atom"}


class ScholarBlocked(RuntimeError):
    pass


FALLBACK_HELP = (
    "Google Scholar refused the automated request (CAPTCHA / rate limit).\n"
    "Fallbacks:\n"
    "  1. Open your profile in a browser, click 'Show more' until all papers are listed,\n"
    "     save the page (Ctrl+S, 'HTML only'), then run:\n"
    "       python research/tools/scholar_fetch.py --html <saved.html>\n"
    "  2. On your profile, tick the select-all box -> Export -> BibTeX, save the file, then run:\n"
    "       python research/tools/scholar_fetch.py --bibtex <scholar.bib>\n"
    "  Or simply retry in a few minutes."
)


# ------------------------------------------------------------------ parsing
def user_id_from_url(url: str) -> str:
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "user" not in qs:
        raise ValueError(f"no user= parameter in {url!r}")
    return qs["user"][0]


def is_blocked(page: str) -> bool:
    return bool(re.search(r"gs_captcha|unusual traffic|id=\"recaptcha|please show you're not a robot", page, re.I))


class _ProfileParser(HTMLParser):
    """Collects publication rows (tr.gsc_a_tr) and the profile name."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self.name = ""
        self.more_disabled = False
        self._row: dict | None = None
        self._field: str | None = None
        self._grays = 0
        self._in_name = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if a.get("id") == "gsc_prf_in":
            self._in_name = True
        if a.get("id") == "gsc_bpf_more" and "disabled" in a:
            self.more_disabled = True
        if tag == "tr" and "gsc_a_tr" in cls:
            self._row = {"title": "", "href": "", "authors": "", "venue": "", "year": "", "citations": ""}
            self._grays = 0
            return
        if self._row is None:
            return
        if tag == "a" and "gsc_a_at" in cls:
            self._field = "title"
            self._row["href"] = a.get("href") or ""
        elif tag == "div" and "gs_gray" in cls:
            self._grays += 1
            self._field = "authors" if self._grays == 1 else "venue"
        elif tag == "span" and "gs_oph" in cls:
            self._field = None  # ", 2026" suffix duplicated from the year column
        elif tag == "a" and "gsc_a_ac" in cls:
            self._field = "citations"
        elif tag == "span" and "gsc_a_h" in cls:
            self._field = "year"

    def handle_endtag(self, tag):
        if self._in_name and tag in ("div", "span"):
            self._in_name = False
        if self._row is None:
            return
        if tag == "tr":
            self.rows.append({k: v.strip() for k, v in self._row.items()})
            self._row = None
            self._field = None
        elif tag in ("a", "div", "span"):
            self._field = None

    def handle_data(self, data):
        if self._in_name:
            self.name += data
        if self._row is not None and self._field:
            self._row[self._field] += data


def parse_profile_html(page: str) -> tuple[list[dict], str, bool]:
    """Return (papers, profile name, more pages possible)."""
    p = _ProfileParser()
    p.feed(page)
    papers = []
    for r in p.rows:
        year = int(r["year"]) if r["year"].isdigit() else None
        cites = int(r["citations"]) if r["citations"].isdigit() else 0
        href = html.unescape(r["href"])
        papers.append({
            "title": r["title"],
            "authors": [a.strip() for a in r["authors"].split(",") if a.strip()],
            "venue": r["venue"].rstrip(", "),
            "year": year,
            "citations": cites,
            "scholar_url": urllib.parse.urljoin(SCHOLAR, href) if href else None,
        })
    return papers, p.name.strip(), not p.more_disabled


def _bib_fields(body: str) -> dict:
    fields = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|\d+)", body):
        val = m.group(2).strip()
        if val[0] in "{\"":
            val = val[1:-1]
        fields[m.group(1).lower()] = re.sub(r"[{}]", "", re.sub(r"\s+", " ", val)).strip()
    return fields


def parse_bibtex(text: str) -> list[dict]:
    papers = []
    for m in re.finditer(r"@(\w+)\s*\{\s*[^,]*,(.*?)\n\}", text, flags=re.S):
        f = _bib_fields(m.group(2))
        if "title" not in f:
            continue
        venue = f.get("journal") or f.get("booktitle") or f.get("publisher") or ""
        papers.append({
            "title": f["title"],
            "authors": [a.strip() for a in f.get("author", "").split(" and ") if a.strip()],
            "venue": venue,
            "year": int(f["year"]) if f.get("year", "").isdigit() else None,
            "citations": None,
            "scholar_url": None,
        })
    return papers


# ------------------------------------------------------------------ network
def _get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise ScholarBlocked(f"HTTP {exc.code} from {urllib.parse.urlparse(url).netloc}") from exc
        raise


def fetch_scholar(user: str, delay: float = 2.5, max_pages: int = 20) -> tuple[list[dict], str]:
    papers: list[dict] = []
    name = ""
    for page in range(max_pages):
        url = (f"{SCHOLAR}/citations?user={urllib.parse.quote(user)}&hl=en&view_op=list_works"
               f"&sortby=pubdate&cstart={page * PAGE_SIZE}&pagesize={PAGE_SIZE}")
        text = _get(url)
        if is_blocked(text):
            raise ScholarBlocked("CAPTCHA page returned")
        rows, pname, more = parse_profile_html(text)
        name = name or pname
        papers.extend(rows)
        if not more or len(rows) < PAGE_SIZE:
            break
        time.sleep(delay)
    return papers, name


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", ATOM):
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", ATOM)).strip()
        if not title or title.lower() == "error":
            continue
        out.append({
            "title": title,
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", ATOM)).strip(),
            "url": e.findtext("a:id", "", ATOM).strip().replace("http://", "https://"),
            "year": int(e.findtext("a:published", "0000", ATOM)[:4]),
        })
    return out


def arxiv_lookup(paper: dict) -> dict | None:
    """Match a paper to arXiv: by explicit arXiv id in the venue, else by exact (normalised) title."""
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", paper.get("venue") or "")
    if m:
        query = {"id_list": m.group(1)}
    else:
        words = norm_title(paper["title"]).split()
        if not words:
            return None
        query = {"search_query": "ti:" + "+AND+ti:".join(words[:12]), "max_results": "5"}
    url = ARXIV_API + "?" + "&".join(f"{k}={v if k == 'search_query' else urllib.parse.quote(v)}" for k, v in query.items())
    entries = None
    for attempt in range(4):  # the arXiv API throttles bursts with 429/503
        try:
            entries = parse_arxiv_feed(_get(url))
            break
        except (urllib.error.URLError, ET.ParseError, ScholarBlocked, TimeoutError) as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    if entries is None:
        print(f"warning: arXiv lookup failed for {paper['title'][:60]!r}: {last}", file=sys.stderr)
        paper["arxiv_lookup_error"] = True
        return None
    target = norm_title(paper["title"])
    for e in entries:
        if norm_title(e["title"]) == target:
            return e
    return None


# ------------------------------------------------------------------ assembly
def finalise(papers: list[dict], recent_years: int, today: dt.date | None = None) -> list[dict]:
    """Sort newest first, assign P# ids and the ``recent`` flag."""
    cutoff = (today or dt.date.today()).year - recent_years + 1
    papers = sorted(papers, key=lambda p: (p.get("year") is None, -(p.get("year") or 0)))
    for i, p in enumerate(papers, 1):
        p["id"] = f"P{i}"
        p["recent"] = None if p.get("year") is None else p["year"] >= cutoff
    return papers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("research/profile/scholar.json"))
    ap.add_argument("--out", type=Path, default=Path("research/profile/scholar_papers.json"))
    ap.add_argument("--html", type=Path, help="parse a saved Scholar profile page instead of fetching")
    ap.add_argument("--bibtex", type=Path, help="parse a Scholar BibTeX export instead of fetching")
    ap.add_argument("--no-arxiv", action="store_true", help="skip arXiv abstract lookup")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    recent_years = int(cfg.get("recent_years", 3))
    name = cfg.get("name", "")
    try:
        if args.bibtex:
            papers, source = parse_bibtex(args.bibtex.read_text(encoding="utf-8")), "bibtex"
        elif args.html:
            text = args.html.read_text(encoding="utf-8", errors="replace")
            if is_blocked(text):
                raise ScholarBlocked("the saved page is a CAPTCHA page")
            papers, name, _ = parse_profile_html(text)
            source = "html"
        else:
            papers, name = fetch_scholar(user_id_from_url(cfg["scholar_url"]))
            source = "google_scholar"
    except ScholarBlocked as exc:
        print(f"error: {exc}\n\n{FALLBACK_HELP}", file=sys.stderr)
        return 2
    if not papers:
        print("error: no papers parsed (is the profile public?)\n\n" + FALLBACK_HELP, file=sys.stderr)
        return 2

    if not args.no_arxiv:
        for p in papers:
            hit = arxiv_lookup(p)
            p["arxiv_url"] = hit["url"] if hit else None
            p["abstract"] = hit["abstract"] if hit else None
            if hit and p.get("year") is None:
                p["year"] = hit["year"]
                p["year_source"] = "arxiv"
            time.sleep(3)  # arXiv API etiquette: at most one request every 3 s
    papers = finalise(papers, recent_years)
    out = {
        "source": source,
        "scholar_url": cfg.get("scholar_url"),
        "name": name,
        "fetched": dt.date.today().isoformat(),
        "recent_years": recent_years,
        "papers": papers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_recent = sum(1 for p in papers if p["recent"])
    n_abs = sum(1 for p in papers if p.get("abstract"))
    n_noyear = sum(1 for p in papers if p["recent"] is None)
    print(f"{name or 'profile'}: {len(papers)} papers, {n_recent} recent (last {recent_years} years), "
          f"{n_abs} with arXiv abstracts, {n_noyear} without a year -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
