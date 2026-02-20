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
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5.2"
BATCH_SIZE = 30
CHECKPOINT_FILE = "enrichment_checkpoint.json"
INPUT_CSV = "airport-code-dataset.csv"
OUTPUT_CSV = "airport-code-dataset-enriched.csv"

PROMPT_TEMPLATE = """For each 3-letter airport code in this list, find any equivalent meanings outside of aviation:
- English words (e.g., BYE = "bye", ANT = "ant")
- Common abbreviations in any domain
- Tech/industry acronyms (e.g., API, CPU, MEM = memory)

Codes: {codes}

Return a JSON object mapping each code to an object with:
- "word": English word if the code spells one, else null
- "abbreviations": list of strings like "domain: meaning" (e.g., "tech: Application Programming Interface")
- "notes": brief summary of notable non-airport meanings, or null if none

Only include codes that have at least one non-airport meaning. For codes with no other meanings, omit them from the response.
Return valid JSON only, no markdown or explanation."""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def load_checkpoint() -> dict:
    path = Path(CHECKPOINT_FILE)
    if not path.exists():
        return {"results": {}, "model": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint: dict) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def call_openai(client: OpenAI, codes: list[str], model: str) -> dict:
    """Call OpenAI API with retry on rate limit."""
    prompt = PROMPT_TEMPLATE.format(codes=", ".join(codes))
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(text)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"Warning: Could not parse JSON for batch {codes[:3]}...: {e}", file=sys.stderr)
            return {}
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt
                print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich airport codes with equivalent meanings via LLM")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI model (default: gpt-5.2)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Codes per API call (default: {BATCH_SIZE})")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    model = args.model
    client = get_client()
    checkpoint = load_checkpoint()
    results: dict[str, str] = checkpoint.get("results", {})

    # Load CSV and collect codes with row order preserved
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or []) + ["meanings"]
        for row in reader:
            rows.append(row)

    # Codes to process (not yet in results)
    codes_to_process: list[str] = []
    code_to_first_row: dict[str, int] = {}
    for i, row in enumerate(rows):
        code = row.get("code", "").strip()
        if not code:
            continue
        if code not in results:
            if code not in code_to_first_row:
                codes_to_process.append(code)
                code_to_first_row[code] = i

    # Process in batches
    total_batches = (len(codes_to_process) + args.batch_size - 1) // args.batch_size
    for b in range(0, len(codes_to_process), args.batch_size):
        batch = codes_to_process[b : b + args.batch_size]
        batch_num = b // args.batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} codes)...", file=sys.stderr)
        batch_results = call_openai(client, batch, model)
        for code in batch:
            key = code.upper()
            data = batch_results.get(code) or batch_results.get(key)
            if data is not None:
                results[key] = json.dumps(data) if isinstance(data, dict) else str(data)
            else:
                results[key] = ""  # No non-airport meanings found
        checkpoint["results"] = results
        checkpoint["model"] = model
        save_checkpoint(checkpoint)

    # Merge results into rows and write output
    for row in rows:
        code = row.get("code", "").strip().upper()
        row["meanings"] = results.get(code, "")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {args.output} with {len(rows)} rows.", file=sys.stderr)


if __name__ == "__main__":
    main()
