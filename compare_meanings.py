#!/usr/bin/env python3
"""Generate an HTML report comparing meanings across provider/prompt columns."""

import argparse
import csv
import html
import itertools
from collections import defaultdict
from pathlib import Path

from config import OUTPUT_CSV

DEFAULT_OUTPUT = "meanings_comparison.html"


def _parse_meanings(cell: str) -> list[str]:
    """Split a semicolon-separated cell into trimmed, non-empty meaning strings."""
    if not cell or not cell.strip():
        return []
    return [m.strip() for m in cell.split(";") if m.strip()]


def _normalize(meaning: str) -> str:
    """Lowercase, strip parentheticals and punctuation for fuzzy token matching."""
    s = meaning.lower().strip()
    # Remove trailing parenthetical like "(software framework)"
    if "(" in s:
        s = s[: s.index("(")].strip()
    return s


def _detect_columns(fieldnames: list[str]) -> list[str]:
    return sorted(c for c in fieldnames if c.startswith("meanings_"))


def _friendly_name(col: str) -> str:
    """Turn 'meanings_openai_false_positive' into 'OpenAI False Positive'."""
    return col.replace("meanings_", "").replace("_", " ").title()


def compute_stats(rows: list[dict], columns: list[str]) -> dict:
    total = len(rows)

    # Per-column sets and counts
    col_codes: dict[str, set[str]] = {c: set() for c in columns}
    col_counts: dict[str, list[int]] = {c: [] for c in columns}
    code_meanings: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for row in rows:
        code = row.get("code", "").strip().upper()
        for col in columns:
            meanings = _parse_meanings(row.get(col, ""))
            if meanings:
                col_codes[col].add(code)
                col_counts[col].append(len(meanings))
                code_meanings[code][col] = meanings

    # --- Per-column summary ---
    per_column = {}
    for col in columns:
        codes = col_codes[col]
        counts = col_counts[col]
        n = len(codes)
        per_column[col] = {
            "count": n,
            "pct": n / total * 100 if total else 0,
            "avg": sum(counts) / n if n else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "total_meanings": sum(counts),
        }

    # --- Pairwise overlap ---
    pairwise = []
    for a, b in itertools.combinations(columns, 2):
        sa, sb = col_codes[a], col_codes[b]
        both = sa & sb
        jaccard = len(both) / len(sa | sb) if (sa | sb) else 0
        pairwise.append({
            "a": a,
            "b": b,
            "both": len(both),
            "only_a": len(sa - sb),
            "only_b": len(sb - sa),
            "jaccard": jaccard,
        })

    # --- All-column overlap ---
    any_set = set()
    all_set: set[str] | None = None
    for col in columns:
        any_set |= col_codes[col]
        if all_set is None:
            all_set = set(col_codes[col])
        else:
            all_set &= col_codes[col]
    all_set = all_set or set()

    overlap = {
        "any": len(any_set),
        "any_pct": len(any_set) / total * 100 if total else 0,
        "all": len(all_set),
        "all_pct": len(all_set) / total * 100 if total else 0,
        "none": total - len(any_set),
        "none_pct": (total - len(any_set)) / total * 100 if total else 0,
    }

    # --- Agreement analysis (pairwise token overlap) ---
    agreement = []
    for a, b in itertools.combinations(columns, 2):
        shared_codes = col_codes[a] & col_codes[b]
        if not shared_codes:
            agreement.append({"a": a, "b": b, "shared_codes": 0, "agree_count": 0, "agree_pct": 0})
            continue
        agree = 0
        for code in shared_codes:
            tokens_a = {_normalize(m) for m in code_meanings[code].get(a, [])}
            tokens_b = {_normalize(m) for m in code_meanings[code].get(b, [])}
            if tokens_a & tokens_b:
                agree += 1
        agreement.append({
            "a": a,
            "b": b,
            "shared_codes": len(shared_codes),
            "agree_count": agree,
            "agree_pct": agree / len(shared_codes) * 100 if shared_codes else 0,
        })

    # --- Top codes by distinct meanings ---
    code_distinct: dict[str, set[str]] = defaultdict(set)
    for code, col_map in code_meanings.items():
        for meanings in col_map.values():
            for m in meanings:
                code_distinct[code].add(_normalize(m))
    top_codes = sorted(code_distinct.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    top_codes_data = []
    for code, _ in top_codes:
        entry = {"code": code, "distinct": len(code_distinct[code])}
        for col in columns:
            entry[col] = code_meanings[code].get(col, [])
        top_codes_data.append(entry)

    return {
        "total": total,
        "columns": columns,
        "per_column": per_column,
        "pairwise": pairwise,
        "overlap": overlap,
        "agreement": agreement,
        "top_codes": top_codes_data,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CARD_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]


def _esc(text: str) -> str:
    return html.escape(str(text))


def _render_html(stats: dict) -> str:
    columns = stats["columns"]
    parts: list[str] = []

    parts.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meanings Comparison Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 2rem; background: #f8fafc; color: #1e293b; line-height: 1.6; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.8rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}
  h2 {{ margin: 2.5rem 0 1rem; font-size: 1.3rem; border-bottom: 2px solid #e2e8f0; padding-bottom: .4rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .card {{ border-radius: 10px; padding: 1.25rem 1.5rem; color: #fff; min-width: 200px; flex: 1; }}
  .card .label {{ font-size: .85rem; opacity: .85; margin-bottom: .25rem; }}
  .card .big {{ font-size: 2rem; font-weight: 700; }}
  .card .detail {{ font-size: .8rem; opacity: .8; margin-top: .35rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ padding: .65rem 1rem; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: .85rem; color: #475569; text-transform: uppercase; letter-spacing: .03em; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  td {{ font-size: .9rem; border-top: 1px solid #e2e8f0; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .tag {{ display: inline-block; background: #e2e8f0; color: #334155; border-radius: 4px;
          padding: .15rem .45rem; font-size: .78rem; margin: .15rem .2rem .15rem 0; }}
  .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .overview-box {{ background: #fff; border-radius: 8px; padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }}
  .overview-box .big {{ font-size: 1.6rem; font-weight: 700; color: #1e293b; }}
  .overview-box .label {{ font-size: .8rem; color: #64748b; }}
</style>
</head>
<body>
<h1>Meanings Comparison Report</h1>
<p class="subtitle">{stats['total']:,} airport codes &middot; {len(columns)} meaning columns</p>
""")

    # --- Per-column cards ---
    parts.append("<h2>Per-Column Summary</h2>\n<div class='cards'>")
    for i, col in enumerate(columns):
        s = stats["per_column"][col]
        color = CARD_COLORS[i % len(CARD_COLORS)]
        parts.append(f"""\
<div class="card" style="background:{color}">
  <div class="label">{_esc(_friendly_name(col))}</div>
  <div class="big">{s['count']:,}</div>
  <div class="detail">{s['pct']:.1f}% of codes &middot; {s['total_meanings']:,} total meanings</div>
  <div class="detail">avg {s['avg']:.1f} &middot; min {s['min']} &middot; max {s['max']} per code</div>
</div>""")
    parts.append("</div>")

    # --- Overall overlap boxes ---
    ov = stats["overlap"]
    parts.append("<h2>Overall Overlap</h2>\n<div class='overview-grid'>")
    for label, key in [("In Any Column", "any"), ("In All Columns", "all"), ("No Meanings", "none")]:
        parts.append(f"""\
<div class="overview-box">
  <div class="big">{ov[key]:,}</div>
  <div class="label">{label} ({ov[key + '_pct']:.1f}%)</div>
</div>""")
    parts.append("</div>")

    # --- Pairwise overlap table ---
    if stats["pairwise"]:
        parts.append("<h2>Pairwise Overlap</h2>\n<table><tr><th>Column A</th><th>Column B</th>"
                     "<th class='num'>Both</th><th class='num'>Only A</th><th class='num'>Only B</th>"
                     "<th class='num'>Jaccard</th></tr>")
        for p in stats["pairwise"]:
            parts.append(f"<tr><td>{_esc(_friendly_name(p['a']))}</td><td>{_esc(_friendly_name(p['b']))}</td>"
                         f"<td class='num'>{p['both']:,}</td><td class='num'>{p['only_a']:,}</td>"
                         f"<td class='num'>{p['only_b']:,}</td><td class='num'>{p['jaccard']:.3f}</td></tr>")
        parts.append("</table>")

    # --- Agreement table ---
    if stats["agreement"]:
        parts.append("<h2>Agreement Analysis</h2>"
                     "<p>For codes present in both columns, how often they share at least one normalized meaning token.</p>"
                     "<table><tr><th>Column A</th><th>Column B</th>"
                     "<th class='num'>Shared Codes</th><th class='num'>Agree</th><th class='num'>Rate</th></tr>")
        for a in stats["agreement"]:
            parts.append(f"<tr><td>{_esc(_friendly_name(a['a']))}</td><td>{_esc(_friendly_name(a['b']))}</td>"
                         f"<td class='num'>{a['shared_codes']:,}</td><td class='num'>{a['agree_count']:,}</td>"
                         f"<td class='num'>{a['agree_pct']:.1f}%</td></tr>")
        parts.append("</table>")

    # --- Top codes ---
    if stats["top_codes"]:
        parts.append("<h2>Top 10 Codes by Distinct Meanings</h2>\n<table><tr><th>Code</th><th class='num'>Distinct</th>")
        for col in columns:
            parts.append(f"<th>{_esc(_friendly_name(col))}</th>")
        parts.append("</tr>")
        for entry in stats["top_codes"]:
            parts.append(f"<tr><td><strong>{_esc(entry['code'])}</strong></td><td class='num'>{entry['distinct']}</td>")
            for col in columns:
                meanings = entry.get(col, [])
                tags = "".join(f"<span class='tag'>{_esc(m)}</span>" for m in meanings)
                parts.append(f"<td>{tags or '&mdash;'}</td>")
            parts.append("</tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML comparison of meanings columns")
    parser.add_argument("--input", default=OUTPUT_CSV, help="Enriched CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Error: {input_path} not found")

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    columns = _detect_columns(fieldnames)
    if not columns:
        raise SystemExit("No meanings_* columns found in the CSV")

    stats = compute_stats(rows, columns)
    html_content = _render_html(stats)

    out = Path(args.output)
    out.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out}  ({len(rows):,} codes, {len(columns)} columns)")


if __name__ == "__main__":
    main()
