#!/usr/bin/env python3
"""
Enrich airport codes with equivalent meanings (words, abbreviations, acronyms)
using OpenAI API. Supports batching, checkpoint resume, and configurable model.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from checkpoint import load_checkpoint, save_checkpoint
from config import BATCH_SIZE, DEFAULT_MODEL, INPUT_CSV, OUTPUT_CSV
from llm import call_openai, get_client

load_dotenv()


def load_rows(input_path: Path) -> tuple[list[dict], list[str]]:
    """Load CSV rows and fieldnames (with meanings column)."""
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or []) + ["meanings"]
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


def run(
    model: str,
    batch_size: int,
    input_path: Path,
    output_path: Path,
) -> None:
    """Run enrichment pipeline."""
    client = get_client()
    checkpoint = load_checkpoint()
    results: dict[str, str] = checkpoint.get("results", {})

    rows, fieldnames = load_rows(input_path)
    codes_to_process = get_codes_to_process(rows, results)

    total_batches = (len(codes_to_process) + batch_size - 1) // batch_size
    effective_model = model

    for b in range(0, len(codes_to_process), batch_size):
        batch = codes_to_process[b : b + batch_size]
        batch_num = b // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} codes)...", file=sys.stderr)

        batch_results, effective_model = call_openai(client, batch, effective_model)

        for code in batch:
            key = code.upper()
            data = batch_results.get(code) or batch_results.get(key)
            results[key] = json.dumps(data) if isinstance(data, dict) else str(data) if data else ""

        checkpoint["results"] = results
        checkpoint["model"] = effective_model
        save_checkpoint(checkpoint)

    for row in rows:
        code = row.get("code", "").strip().upper()
        row["meanings"] = results.get(code, "")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {output_path} with {len(rows)} rows.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich airport codes with equivalent meanings via LLM")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Codes per API call")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run(
        model=args.model,
        batch_size=args.batch_size,
        input_path=input_path,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
