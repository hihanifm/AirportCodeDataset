"""OpenAI client and API call logic with fallback models."""

import json
import os
import sys
import time

from openai import OpenAI

from config import FALLBACK_MODELS, PROMPT_TEMPLATE


def get_client() -> OpenAI:
    """Create OpenAI client; exits if API key not set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def _is_model_error(e: Exception) -> bool:
    """Check if error is due to invalid/unavailable model (try fallback)."""
    msg = str(e).lower()
    if "429" in msg or "rate" in msg:
        return False  # Rate limit: retry same model
    if getattr(e, "status_code", None) in (404, 400):
        return True
    model_phrases = ("model", "does not exist", "invalid model", "not found", "not available")
    return any(p in msg for p in model_phrases)


def call_openai(
    client: OpenAI,
    codes: list[str],
    model: str,
    fallbacks: list[str] | None = None,
) -> tuple[dict, str]:
    """
    Call OpenAI API with retry on rate limit and fallback on model errors.
    Returns (result, model_used).
    """
    fallback_list = fallbacks if fallbacks is not None else FALLBACK_MODELS
    models_to_try = [model] + [m for m in fallback_list if m != model]
    prompt = PROMPT_TEMPLATE.format(codes=", ".join(codes))
    last_error: Exception | None = None

    for current_model in models_to_try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                return (json.loads(text), current_model)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"Warning: Could not parse JSON for batch {codes[:3]}...: {e}", file=sys.stderr)
                return ({}, current_model)
            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2**attempt
                    print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if _is_model_error(e):
                    print(f"Model {current_model} failed: {e}. Trying fallback...", file=sys.stderr)
                    break  # Try next model
                raise

    raise last_error or RuntimeError("All models failed")
