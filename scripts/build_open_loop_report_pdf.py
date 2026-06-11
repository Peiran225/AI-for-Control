"""Build HTML/PDF versions of the open-loop u(t) reports."""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "pdf_assets"


CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body {
  font-family: Arial, Helvetica, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
  color: #1f2933;
  line-height: 1.45;
  font-size: 11pt;
}
h1 { font-size: 24pt; margin: 0 0 18px 0; color: #111827; }
h2 { font-size: 16pt; margin: 24px 0 10px 0; color: #111827; border-bottom: 1px solid #d8dee9; padding-bottom: 4px; }
p { margin: 8px 0; }
.math-block {
  font-family: "Times New Roman", "Noto Serif", serif;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  background: #f7f8fb;
  border: 1px solid #e1e5ee;
  border-radius: 6px;
  padding: 8px 10px;
  margin: 10px 0;
  text-align: center;
  color: #111827;
}
.math-inline {
  font-family: "Times New Roman", "Noto Serif", serif;
  color: #111827;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 16px 0;
  font-size: 9.5pt;
}
th, td {
  border: 1px solid #cfd6e4;
  padding: 5px 7px;
  vertical-align: top;
}
th { background: #eef2f7; font-weight: 700; }
td.num, th.num { text-align: right; }
figure { margin: 13px 0 18px 0; page-break-inside: avoid; width: 100%; }
img { display: block; width: 100%; height: auto; margin: 0 auto; border: 1px solid #d7dde8; }
figcaption { text-align: center; color: #52606d; font-size: 9pt; margin-top: 5px; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
"""


def inline_markup(text: str, base_dir: Path) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\\\((.*?)\\\)", r"<span class=\"math-inline\">\1</span>", text)
    return text


def image_html(line: str, base_dir: Path) -> str | None:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not match:
        return None
    alt, src = match.groups()
    img_path = (base_dir / src).resolve()
    ASSETS.mkdir(exist_ok=True)
    asset_path = ASSETS / img_path.name
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        max_width = 650
        if im.width > max_width:
            new_height = max(1, int(round(im.height * max_width / im.width)))
            im = im.resize((max_width, new_height), Image.LANCZOS)
        im.save(asset_path)
    return (
        "<figure>"
        f"<img src=\"{asset_path.resolve().as_uri()}\" alt=\"{html.escape(alt)}\">"
        "</figure>"
    )


def parse_table(lines: list[str], start: int, base_dir: Path) -> tuple[str, int]:
    header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
    i = start + 2
    rows: list[list[str]] = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
        i += 1
    html_rows = ["<table><thead><tr>"]
    for cell in header:
        cls = " class=\"num\"" if re.search(r"value|J|gap|loss|epoch|component", cell, re.I) else ""
        html_rows.append(f"<th{cls}>{inline_markup(cell, base_dir)}</th>")
    html_rows.append("</tr></thead><tbody>")
    for row in rows:
        html_rows.append("<tr>")
        for cell in row:
            cls = " class=\"num\"" if re.fullmatch(r"[-+0-9., ]+", cell) else ""
            html_rows.append(f"<td{cls}>{inline_markup(cell, base_dir)}</td>")
        html_rows.append("</tr>")
    html_rows.append("</tbody></table>")
    return "".join(html_rows), i


def md_to_html(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    in_math = False
    math_lines: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{inline_markup(' '.join(para), path.parent)}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$":
            flush_para()
            if in_math:
                out.append(f"<div class=\"math-block\">{html.escape(chr(10).join(math_lines))}</div>")
                math_lines.clear()
                in_math = False
            else:
                in_math = True
            i += 1
            continue

        if in_math:
            math_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_para()
            out.append(f"<h1>{inline_markup(stripped[2:], path.parent)}</h1>")
            i += 1
            continue

        if stripped.startswith("## "):
            flush_para()
            out.append(f"<h2>{inline_markup(stripped[3:], path.parent)}</h2>")
            i += 1
            continue

        img = image_html(stripped, path.parent)
        if img:
            flush_para()
            out.append(img)
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            flush_para()
            table, i = parse_table(lines, i, path.parent)
            out.append(table)
            continue

        para.append(stripped)
        i += 1

    flush_para()
    return "\n".join(out)


def build_one(md_name: str) -> Path:
    md_path = REPORTS / md_name
    html_path = md_path.with_suffix(".html")
    body = md_to_html(md_path)
    html_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(md_path.stem)}</title><style>{CSS}</style></head><body>"
        f"{body}</body></html>",
        encoding="utf-8",
    )
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(REPORTS), str(html_path)],
        check=True,
        cwd=str(ROOT),
    )
    return html_path.with_suffix(".pdf")


def main() -> None:
    for name in ["open_loop_ut_reproduction_report.md", "open_loop_ut_reproduction_report_zh.md"]:
        pdf = build_one(name)
        print(pdf)


if __name__ == "__main__":
    main()
