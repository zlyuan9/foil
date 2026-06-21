"""
Mirror example generation using Gemini.
Given false positive texts (human text the model thinks is AI),
generate actual AI versions of similar content.

Two-step batched process:
1. Batch reverse-engineer prompts from original texts (10 per API call)
2. Batch generate new texts from those prompts (10 per API call)
"""

import os
import time
import json
import random

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"
BATCH_SIZE = 10
MAX_RETRIES = 3


def _call_gemini(contents: str, retries: int = MAX_RETRIES) -> str:
    """Call Gemini with retry logic for rate limits and transient errors."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
            )
            if response.text is None:
                raise ValueError("Empty response from Gemini (likely safety filter)")
            return response.text
        except genai_errors.ClientError as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait = 2 ** (attempt + 1) * 5
                print(f"    Rate limited. Waiting {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            elif "400" in error_msg and "API key" in error_msg:
                raise RuntimeError(f"API key invalid: {e}") from e
            else:
                wait = 2 ** attempt
                print(f"    Client error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        except genai_errors.ServerError as e:
            wait = 2 ** (attempt + 1) * 3
            print(f"    Server error: {e}. Retrying in {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    Unexpected error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {retries} retries")


def batch_generate_prompts(texts: list[str]) -> list[str]:
    """Reverse-engineer prompts for a batch of texts in one API call."""
    numbered = "\n\n".join(
        f"[{i+1}]\n\"\"\"{t[:1000]}\"\"\"" for i, t in enumerate(texts)
    )
    response_text = _call_gemini(
        f"""For each of the following {len(texts)} texts, write a short prompt (1-2 sentences) that someone might use to ask an AI to generate something very similar. Focus on the topic, style, tone, and format.

{numbered}

Respond with ONLY the prompts, one per line, numbered [1] through [{len(texts)}]. No other text."""
    )

    lines = response_text.strip().split("\n")
    prompts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbering like [1], 1., 1), etc.
        for prefix in [f"[{len(prompts)+1}]", f"{len(prompts)+1}.", f"{len(prompts)+1})"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            prompts.append(line)

    # Pad if we got fewer prompts than texts
    while len(prompts) < len(texts):
        prompts.append(f"Write a text similar to: {texts[len(prompts)][:200]}")

    return prompts[:len(texts)]


def batch_generate_mirrors(prompts: list[str]) -> list[str]:
    """Generate AI text for a batch of prompts in one API call."""
    numbered = "\n\n".join(
        f"[{i+1}] {p}" for i, p in enumerate(prompts)
    )
    response_text = _call_gemini(
        f"""Write {len(prompts)} separate texts based on the following prompts. Each response should be 100-300 words. Separate each response with "---".

{numbered}

Write each response naturally, as if you're fulfilling each prompt independently. Separate with "---"."""
    )

    parts = response_text.split("---")
    results = [p.strip() for p in parts if len(p.strip()) > 50]

    # Pad if we got fewer results
    while len(results) < len(prompts):
        results.append("")

    return results[:len(prompts)]


def generate_mirrors(
    false_positive_texts: list[str],
    max_mirrors: int = 500,
    delay: float = 1.0,
) -> list[dict]:
    """
    Generate mirror examples from false positive texts using batched API calls.
    Returns list of dicts with keys: [text, label, source, domain]
    """
    if len(false_positive_texts) > max_mirrors:
        false_positive_texts = random.sample(false_positive_texts, max_mirrors)

    mirrors = []
    total = len(false_positive_texts)

    for i in range(0, total, BATCH_SIZE):
        batch_texts = false_positive_texts[i : i + BATCH_SIZE]

        try:
            # Step 1: Batch reverse-engineer prompts
            prompts = batch_generate_prompts(batch_texts)
            time.sleep(delay)

            # Step 2: Batch generate mirror texts
            mirror_texts = batch_generate_mirrors(prompts)
            time.sleep(delay)

            for text in mirror_texts:
                if len(text.strip()) > 100:
                    mirrors.append({
                        "text": text.strip(),
                        "label": 1,
                        "source": "mirror",
                        "domain": "mirror",
                    })

            print(f"  Batch {i//BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1)//BATCH_SIZE}: "
                  f"{len(mirrors)} mirrors generated so far")

        except Exception as e:
            print(f"  Error on batch starting at {i}: {e}")
            time.sleep(3)
            continue

    return mirrors


if __name__ == "__main__":
    test_texts = [
        "Four stars. Only gets better as the years go by. Beautiful meditation on finding new meaning in life. Great, tactile feeling direction that brings every scene to life so vividly. Deliriously funny.",
        "The restaurant was absolutely packed on a Friday night but the service was still impeccable. Our waiter remembered every single order without writing anything down. The risotto was creamy and perfectly seasoned.",
        "I've been using this laptop for about six months now and the battery life has noticeably degraded. Customer support was unhelpful and kept redirecting me to different departments.",
    ]

    print(f"Testing batched mirror generation with {len(test_texts)} texts...\n")

    print("Step 1: Batch reverse-engineering prompts...")
    prompts = batch_generate_prompts(test_texts)
    for i, p in enumerate(prompts):
        print(f"  [{i+1}] {p}")

    print("\nStep 2: Batch generating mirrors...")
    mirror_texts = batch_generate_mirrors(prompts)
    for i, m in enumerate(mirror_texts):
        print(f"\n  [{i+1}] {m[:200]}...")
