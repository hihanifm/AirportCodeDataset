#!/usr/bin/env python3
"""
Enrich airport codes with equivalent meanings (words, abbreviations, acronyms)
using LLM providers (OpenAI, Gemini). Supports batching, checkpoint resume,
configurable model, and per-provider columns in the output CSV.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from checkpoint import load_checkpoint, save_checkpoint
from config import BATCH_SIZE, INPUT_CSV, OUTPUT_CSV, PROMPTS, PROVIDER_DEFAULTS
from llm import call_llm, get_client

load_dotenv()


def _to_semicolon_separated(data: dict | None) -> str:
    """Extract semicolon-separated values from LLM response, stripping domain prefixes."""
    if not data or not isinstance(data, dict):
        return ""
    values = []
    if data.get("word"):
        values.append(str(data["word"]))
    for abbr in data.get("abbreviations") or []:
        s = abbr if isinstance(abbr, str) else str(abbr)
        if ": " in s:
            s = s.split(": ", 1)[1]
        if s and s not in values:
            values.append(s)
    return "; ".join(values)


def _load_existing_output(output_path: Path) -> dict[str, dict]:
    """Load existing enriched CSV rows keyed by code (for merging columns)."""
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("code", "").strip(): dict(row) for row in reader}


def load_rows(input_path: Path) -> tuple[list[dict], list[str]]:
    """Load CSV rows and base fieldnames."""
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def get_codes_to_process(rows: list[dict], results: dict[str, str]) -> list[str]:
    """Return unique codes not yet in results (preserving first-seen order)."""
    seen: set[str] = set()
    codes: list[str] = []
    for row in rows:
        code = row.get("code", "").strip().upper()
        if not code or code in results or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _make_column_and_key(provider: str, prompt_name: str) -> tuple[str, str]:
    """Derive CSV column name and checkpoint key from provider + prompt."""
    base = PROVIDER_DEFAULTS[provider]["column"]
    if prompt_name == "generic":
        return base, provider
    suffix = prompt_name.replace("-", "_")
    return f"{base}_{suffix}", f"{provider}_{suffix}"


def run(
    provider: str,
    model: str,
    batch_size: int,
    input_path: Path,
    output_path: Path,
    prompt_template: str | None = None,
    prompt_name: str = "generic",
) -> None:
    """Run enrichment pipeline for a given provider."""
    column, checkpoint_key = _make_column_and_key(provider, prompt_name)
    client = get_client(provider)
    checkpoint = load_checkpoint(checkpoint_key)
    results: dict[str, str] = checkpoint.get("results", {})

    rows, base_fieldnames = load_rows(input_path)
    codes_to_process = get_codes_to_process(rows, results)

    total_batches = (len(codes_to_process) + batch_size - 1) // batch_size
    effective_model = model

    for b in range(0, len(codes_to_process), batch_size):
        batch = codes_to_process[b : b + batch_size]
        batch_num = b // batch_size + 1
        if not sys.stderr.isatty():
            print(f"[{provider}] Processing batch {batch_num}/{total_batches} ({len(batch)} codes)...", file=sys.stderr)

        batch_results, effective_model = call_llm(client, batch, effective_model, provider, prompt_template=prompt_template)

        for code in batch:
            key = code.upper()
            data = batch_results.get(code) or batch_results.get(key)
            value = _to_semicolon_separated(data) if isinstance(data, dict) else (str(data) if data else "")
            results[key] = value
            print(f"  {code} → {value or '—'}", file=sys.stderr)

        checkpoint["results"] = results
        checkpoint["model"] = effective_model
        save_checkpoint(checkpoint_key, checkpoint)

        # Write partial output after each batch
        _write_output(rows, base_fieldnames, results, column, output_path)

    if sys.stderr.isatty():
        print(file=sys.stderr)
    print(f"[{provider}] Done. Wrote {output_path} with {len(rows)} rows.", file=sys.stderr)


def _write_output(
    rows: list[dict],
    base_fieldnames: list[str],
    results: dict[str, str],
    column: str,
    output_path: Path,
) -> None:
    """Write enriched CSV, preserving existing columns from other runs."""
    existing = _load_existing_output(output_path)

    # Discover all enrichment columns: from config + from existing CSV
    known_cols: set[str] = set()
    for conf in PROVIDER_DEFAULTS.values():
        known_cols.add(conf["column"])
    if existing:
        sample = next(iter(existing.values()), {})
        for key in sample:
            if key.startswith("meanings_"):
                known_cols.add(key)
    known_cols.add(column)

    fieldnames = list(base_fieldnames)
    for col in sorted(known_cols):
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        code = row.get("code", "").strip()
        code_upper = code.upper()
        # Preserve other columns from existing output
        if code in existing:
            for col in known_cols:
                if col != column and col in existing[code]:
                    row[col] = existing[code][col]
        # Set current provider column
        raw = results.get(code_upper, "")
        if raw.startswith("{"):
            try:
                raw = _to_semicolon_separated(json.loads(raw))
            except json.JSONDecodeError:
                pass
        row[column] = raw

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich airport codes with equivalent meanings via LLM")
    parser.add_argument("--provider", default="openai", choices=list(PROVIDER_DEFAULTS.keys()), help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None, help="Model name (default depends on provider)")
    parser.add_argument("--prompt", default="generic", choices=list(PROMPTS.keys()), help="Prompt variant (default: generic)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Codes per API call")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    provider = args.provider
    model = args.model or os.getenv("LLM_MODEL") or PROVIDER_DEFAULTS[provider]["model"]
    prompt_template = PROMPTS[args.prompt]

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run(
        provider=provider,
        model=model,
        batch_size=args.batch_size,
        input_path=input_path,
        output_path=Path(args.output),
        prompt_template=prompt_template,
        prompt_name=args.prompt,
    )


if __name__ == "__main__":
    main()
