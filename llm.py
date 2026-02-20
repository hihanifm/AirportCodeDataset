"""LLM client and API call logic with fallback models. Supports OpenAI and Gemini."""

import json
import os
import sys
import time

from config import FALLBACK_MODELS, GEMINI_FALLBACK_MODELS, PROMPT_TEMPLATE


def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def _get_gemini_client():
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


def get_client(provider: str):
    """Create LLM client for the given provider."""
    if provider == "openai":
        return _get_openai_client()
    elif provider == "gemini":
        return _get_gemini_client()
    else:
        print(f"Error: Unknown provider '{provider}'. Use 'openai' or 'gemini'.", file=sys.stderr)
        sys.exit(1)


def _is_model_error(e: Exception) -> bool:
    """Check if error is due to invalid/unavailable model (try fallback)."""
    msg = str(e).lower()
    if "429" in msg or "rate" in msg:
        return False
    if getattr(e, "status_code", None) in (404, 400):
        return True
    model_phrases = ("model", "does not exist", "invalid model", "not found", "not available")
    return any(p in msg for p in model_phrases)


def _parse_response_text(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _call_openai(client, codes: list[str], model: str, fallbacks: list[str], prompt_template: str | None = None) -> tuple[dict, str]:
    models_to_try = [model] + [m for m in fallbacks if m != model]
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.format(codes=", ".join(codes))
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
                return (_parse_response_text(response.choices[0].message.content), current_model)
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
                    break
                raise

    raise last_error or RuntimeError("All OpenAI models failed")


def _call_gemini(client, codes: list[str], model: str, fallbacks: list[str], prompt_template: str | None = None) -> tuple[dict, str]:
    models_to_try = [model] + [m for m in fallbacks if m != model]
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.format(codes=", ".join(codes))
    last_error: Exception | None = None

    for current_model in models_to_try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                )
                return (_parse_response_text(response.text), current_model)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"Warning: Could not parse JSON for batch {codes[:3]}...: {e}", file=sys.stderr)
                return ({}, current_model)
            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower() or "resource_exhausted" in str(e).lower():
                    wait = 2**attempt
                    print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if _is_model_error(e):
                    print(f"Model {current_model} failed: {e}. Trying fallback...", file=sys.stderr)
                    break
                raise

    raise last_error or RuntimeError("All Gemini models failed")


def call_llm(
    client,
    codes: list[str],
    model: str,
    provider: str,
    fallbacks: list[str] | None = None,
    prompt_template: str | None = None,
) -> tuple[dict, str]:
    """
    Call LLM API with retry and fallback. Returns (result, model_used).
    """
    if provider == "openai":
        fb = fallbacks if fallbacks is not None else FALLBACK_MODELS
        return _call_openai(client, codes, model, fb, prompt_template)
    elif provider == "gemini":
        fb = fallbacks if fallbacks is not None else GEMINI_FALLBACK_MODELS
        return _call_gemini(client, codes, model, fb, prompt_template)
    else:
        raise ValueError(f"Unknown provider: {provider}")
