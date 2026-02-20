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

PROMPTS = {
    "generic": """For each 3-letter code in this list, identify ONLY well-established, factual meanings outside of aviation. Do NOT invent or guess meanings. Only include meanings that are widely recognized and verifiable.

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
Return valid JSON only, no markdown or explanation.""",

    "false-positive": """Context: We are building a smartphone feature that detects IATA airport codes in emails and messages to offer travel hints (e.g., showing flight info when "JFK" appears). The problem is false positives — many 3-letter airport codes are also common English words (e.g., "BYE", "AND", "THE") or widely used abbreviations (e.g., "API", "CPU", "MEM"). We need to catalog these collisions so our model can distinguish between airport references and everyday language.

For each 3-letter code below, identify ONLY well-established, factual non-aviation meanings. These are the meanings that would cause a false positive if someone typed them in a message and our system mistakenly treated them as an airport code.

Categories to check:
- Common English words (e.g., BYE = "bye", ANT = "ant", MEN = "men", THE = "the")
- Widely used abbreviations and acronyms in technology, medicine, business, education, or everyday life (e.g., API = Application Programming Interface, CPU = Central Processing Unit, MBA = Master of Business Administration)
- Common slang, texting shorthand, or internet abbreviations (e.g., LOL, BRB, OMG)
- Country/currency/unit codes that people use in messages (e.g., USD, GBP, KGS)

Codes: {codes}

Return a JSON object mapping each code to an object with:
- "word": English dictionary word if the code exactly spells one (case-insensitive), else null
- "abbreviations": list of strings like "domain: meaning" — only widely known, established meanings that a regular person might type in a message or email
- "notes": null (do not add notes or commentary)

Strict rules:
- Do NOT make up or speculate about meanings
- Do NOT include obscure, niche, or domain-specific jargon that ordinary people would never use in messages
- Only include codes that have at least one real, verifiable non-aviation meaning
- Omit codes with no well-known non-airport meaning
- Think about it from the perspective of: "Would a normal person type this in an email or text message meaning something other than an airport?"
Return valid JSON only, no markdown or explanation.""",
}

DEFAULT_PROMPT = "generic"
PROMPT_TEMPLATE = PROMPTS[DEFAULT_PROMPT]
