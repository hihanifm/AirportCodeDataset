"""Configuration constants for airport code enrichment."""

# OpenAI
DEFAULT_MODEL = "gpt-5.2"
FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]

# Gemini
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro"]

# Provider defaults: provider -> (default_model, column_name)
PROVIDER_DEFAULTS = {
    "openai": {"model": DEFAULT_MODEL, "column": "meanings_openai"},
    "gemini": {"model": DEFAULT_GEMINI_MODEL, "column": "meanings_gemini"},
}

BATCH_SIZE = 30
CHECKPOINT_FILE = "enrichment_checkpoint.json"
INPUT_CSV = "airport-code-dataset.csv"
OUTPUT_CSV = "airport-code-dataset-enriched.csv"

PROMPT_TEMPLATE = """For each 3-letter code in this list, identify ONLY well-established, factual meanings outside of aviation. Do NOT invent or guess meanings. Only include meanings that are widely recognized and verifiable.

Categories:
- English dictionary words (e.g., BYE = "bye", ANT = "ant", MEN = "men")
- Officially recognized abbreviations and acronyms (e.g., API = Application Programming Interface, CPU = Central Processing Unit)

Codes: {codes}

Return a JSON object mapping each code to an object with:
- "word": English dictionary word if the code exactly spells one (case-insensitive), else null
- "abbreviations": list of strings like "domain: meaning" — only include widely known, established abbreviations
- "notes": null (do not add notes or commentary)

Strict rules:
- Do NOT make up or speculate about meanings
- Do NOT include obscure or niche abbreviations
- Only include codes that have at least one real, verifiable meaning
- Omit codes with no well-known non-airport meaning
Return valid JSON only, no markdown or explanation."""
