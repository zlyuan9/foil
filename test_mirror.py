"""
Test cases for mirror generation to validate before running on Modal.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

from mirror_gen import (
    _call_gemini,
    batch_generate_prompts,
    batch_generate_mirrors,
    generate_mirrors,
)


def test_gemini_connection():
    """Test that we can reach Gemini at all."""
    print("TEST: Gemini connection...")
    response = _call_gemini("Say 'hello' in one word.")
    assert response is not None
    assert len(response.strip()) > 0
    print(f"  PASS: got '{response.strip()}'")


def test_batch_generate_prompts_single():
    """Test prompt generation with a single text."""
    print("TEST: Batch generate prompts (1 text)...")
    texts = ["The pizza was incredible. Thin crust, perfect char, fresh mozzarella. Best I've had in years."]
    prompts = batch_generate_prompts(texts)
    assert len(prompts) == 1, f"Expected 1 prompt, got {len(prompts)}"
    assert len(prompts[0]) > 10, f"Prompt too short: '{prompts[0]}'"
    print(f"  PASS: '{prompts[0][:80]}...'")


def test_batch_generate_prompts_multiple():
    """Test prompt generation with multiple texts — verifies we get the right count back."""
    print("TEST: Batch generate prompts (5 texts)...")
    texts = [
        "The sunset painted the sky in shades of orange and purple.",
        "I've been using Python for about three years now and finally understand decorators.",
        "Customer service was rude and unhelpful. Never shopping here again.",
        "The algorithm runs in O(n log n) time using a divide and conquer approach.",
        "My grandmother's recipe calls for two cups of flour and a pinch of salt.",
    ]
    prompts = batch_generate_prompts(texts)
    assert len(prompts) == 5, f"Expected 5 prompts, got {len(prompts)}"
    for i, p in enumerate(prompts):
        assert len(p) > 10, f"Prompt {i} too short: '{p}'"
    print(f"  PASS: got {len(prompts)} prompts")
    for i, p in enumerate(prompts):
        print(f"    [{i+1}] {p[:70]}")


def test_batch_generate_mirrors_single():
    """Test mirror text generation from a single prompt."""
    print("TEST: Batch generate mirrors (1 prompt)...")
    prompts = ["Write a short positive review of a pizza restaurant, focusing on the crust."]
    mirrors = batch_generate_mirrors(prompts)
    assert len(mirrors) >= 1, f"Expected at least 1 mirror, got {len(mirrors)}"
    assert len(mirrors[0]) > 50, f"Mirror too short: '{mirrors[0][:50]}'"
    print(f"  PASS: got {len(mirrors)} mirror(s), first is {len(mirrors[0])} chars")


def test_batch_generate_mirrors_multiple():
    """Test mirror generation with multiple prompts — verifies separator parsing."""
    print("TEST: Batch generate mirrors (3 prompts)...")
    prompts = [
        "Write a one-paragraph restaurant review praising the ambiance.",
        "Write a short complaint about slow delivery times.",
        "Write a brief movie review that's mostly positive but mentions one flaw.",
    ]
    mirrors = batch_generate_mirrors(prompts)
    assert len(mirrors) >= 2, f"Expected at least 2 mirrors, got {len(mirrors)}"
    print(f"  PASS: got {len(mirrors)} mirrors")
    for i, m in enumerate(mirrors):
        print(f"    [{i+1}] ({len(m)} chars) {m[:60]}...")


def test_end_to_end_generate_mirrors():
    """Test the full generate_mirrors function with a small batch."""
    print("TEST: End-to-end generate_mirrors (3 texts, max_mirrors=3)...")
    texts = [
        "Absolutely loved this hotel. The rooftop pool was stunning and the staff went above and beyond to make our anniversary special. Would recommend to anyone visiting downtown.",
        "The new MacBook Pro is a solid upgrade but I'm not sure it justifies the price increase. Battery life is better but the keyboard still feels mushy to me.",
        "Professor Smith's lecture on quantum entanglement was mind-blowing. She explained Bell's theorem in a way that finally clicked for me after years of confusion.",
    ]
    mirrors = generate_mirrors(texts, max_mirrors=3, delay=0.5)
    assert len(mirrors) > 0, "Expected at least 1 mirror"
    for m in mirrors:
        assert m["label"] == 1, f"Mirror should have label=1, got {m['label']}"
        assert m["source"] == "mirror"
        assert len(m["text"]) > 100, f"Mirror text too short: {len(m['text'])} chars"
    print(f"  PASS: got {len(mirrors)} mirrors, all valid")


def test_empty_input():
    """Test that empty input doesn't crash."""
    print("TEST: Empty input...")
    mirrors = generate_mirrors([], max_mirrors=5)
    assert mirrors == [], f"Expected empty list, got {len(mirrors)} items"
    print("  PASS: returned empty list")


if __name__ == "__main__":
    tests = [
        test_gemini_connection,
        test_batch_generate_prompts_single,
        test_batch_generate_prompts_multiple,
        test_batch_generate_mirrors_single,
        test_batch_generate_mirrors_multiple,
        test_end_to_end_generate_mirrors,
        test_empty_input,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
        print()

    print(f"{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
