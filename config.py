"""Configuration constants for airport code enrichment."""

DEFAULT_MODEL = "gpt-5.2"
FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
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
