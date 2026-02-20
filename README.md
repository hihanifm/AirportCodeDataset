# Airport Code Dataset - Meaning Enrichment

Enriches 9,766 IATA airport codes with equivalent non-aviation meanings (English words, common abbreviations, tech acronyms, texting shorthand) using LLM APIs.

**Use case**: A smartphone feature that detects airport codes in emails and messages to show travel hints. Many 3-letter codes collide with everyday words and abbreviations (`BYE`, `AND`, `API`, `LOL`), causing false positives. This dataset catalogs those collisions so a model can distinguish airport references from ordinary language.

## Output

`airport-code-dataset-enriched.csv` adds provider-specific columns to the original dataset:

| Column | Source |
|--------|--------|
| `meanings_openai` | OpenAI (gpt-5.2) |
| `meanings_gemini` | Google Gemini (gemini-2.5-pro) |

Values are semicolon-separated. Empty means no non-aviation meaning found.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your API keys
```

## Usage

```bash
# OpenAI (default)
python enrich_airport_codes.py

# Gemini
python enrich_airport_codes.py --provider gemini

# Custom model
python enrich_airport_codes.py --provider openai --model gpt-4o

# Background (survives terminal close and screen off)
./run_enrich.sh                        # OpenAI
./run_enrich.sh --provider gemini      # Gemini

# Stop
./stop_enrich.sh

# Watch progress
tail -f enrich.log
```

## Features

- **Multi-provider**: OpenAI and Google Gemini, each with its own CSV column
- **Checkpoint resume**: Interrupted runs continue where they left off
- **Fallback models**: Automatically tries alternative models if the primary fails
- **Batch processing**: 30 codes per API call to reduce cost and rate limits

## Files

| File | Purpose |
|------|---------|
| `enrich_airport_codes.py` | Main script |
| `llm.py` | LLM client logic (OpenAI, Gemini) |
| `config.py` | Constants, models, prompt |
| `checkpoint.py` | Resume support |
| `run_enrich.sh` | Background runner |
| `stop_enrich.sh` | Stop background process |
