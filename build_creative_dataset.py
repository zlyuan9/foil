"""
Build creative writing dataset:
1. Pull human creative writing from WritingPrompts (already cached)
2. Generate AI creative writing via Gemini Flash
3. Combine, tokenize, merge into active training pool
"""

import os
import time
import json
import random

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from google import genai
from google.genai import errors as genai_errors

load_dotenv(override=True)

DATA_DIR = "data"
MODEL_NAME = "microsoft/deberta-v3-base"
TARGET_HUMAN = 3000
TARGET_AI = 3000
BATCH_SIZE = 5


# ============================================================
# Step 1: Human creative writing from WritingPrompts
# ============================================================
def get_writing_prompts(n=TARGET_HUMAN):
    print("Loading WritingPrompts (human creative writing)...")
    rows = []

    # Test split is cached (15k stories)
    path = hf_hub_download(
        "euclaise/writingprompts",
        "data/test-00000-of-00001-16503b0c26ed00c6.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        story = row["story"].strip()
        if 200 < len(story) < 8000:
            rows.append(story)

    # Also grab train split if we need more
    if len(rows) < n:
        for split_file in [
            "data/train-00000-of-00002-105e07cb0d199464.parquet",
            "data/train-00001-of-00002-4fdb982c11056472.parquet",
        ]:
            try:
                print(f"  Downloading {split_file}...")
                path = hf_hub_download(
                    "euclaise/writingprompts", split_file,
                    repo_type="dataset",
                )
                df = pd.read_parquet(path)
                for _, row in df.iterrows():
                    story = row["story"].strip()
                    if 200 < len(story) < 8000:
                        rows.append(story)
                    if len(rows) >= n * 2:
                        break
            except Exception as e:
                print(f"  Failed to get {split_file}: {e}")
            if len(rows) >= n * 2:
                break

    random.shuffle(rows)
    rows = rows[:n]
    print(f"  Got {len(rows)} human creative writing samples")
    return rows


# ============================================================
# Step 2: AI creative writing via Gemini Flash
# ============================================================
CREATIVE_TOPICS = [
    "a detective solving their last case", "first contact with aliens",
    "a letter never sent", "the last day of summer", "a stranger on a train",
    "waking up in someone else's body", "a secret room in an old house",
    "the day the music stopped", "a conversation between ghosts",
    "finding a message in a bottle", "a heist gone wrong",
    "a chef's worst night", "parallel universes colliding",
    "an astronaut's diary entry", "a forbidden library",
    "a deal with the devil", "the view from a lighthouse",
    "an unreliable narrator at a dinner party", "time travel gone wrong",
    "a love story told backwards", "the last bookshop in the city",
    "a storm at sea", "a lost civilization",
    "a musician's final performance", "a garden that grows memories",
    "a map to nowhere", "a rival's redemption",
    "a king disguised as a beggar", "the inheritance nobody wanted",
    "a photograph that changes everything", "the last train home",
    "an impossible choice", "a world without color",
    "the neighbor's secret", "a forgotten promise",
    "the space between heartbeats", "a city that never sleeps",
    "the wrong door", "a voice from the past",
    "the price of immortality", "a bridge between worlds",
]

STYLES = ["literary fiction", "thriller", "mystery", "sci-fi", "fantasy",
          "magical realism", "slice-of-life", "horror", "romance", "historical"]


def generate_ai_creative(n=TARGET_AI):
    print(f"\nGenerating {n} AI creative writing samples via Gemini Flash...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ERROR: No GEMINI_API_KEY in .env")
        return []

    client = genai.Client(api_key=api_key)
    MODEL = "gemini-2.5-flash"

    def call_gemini(contents, retries=3):
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=MODEL, contents=contents
                )
                if response.text is None:
                    raise ValueError("Empty response")
                return response.text
            except genai_errors.ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 2 ** (attempt + 1) * 10
                    print(f"    Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif "API key" in str(e):
                    raise
                else:
                    time.sleep(2 ** attempt * 2)
            except genai_errors.ServerError:
                time.sleep(2 ** (attempt + 1) * 5)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)
        raise RuntimeError("Gemini failed after retries")

    # Resume support
    progress_path = os.path.join(DATA_DIR, "ai_creative_progress.jsonl")
    existing = []
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            for line in f:
                existing.append(json.loads(line)["text"])
        print(f"  Resuming from {len(existing)} existing samples")

    texts = list(existing)
    batches_needed = (n - len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(batches_needed):
        if len(texts) >= n:
            break

        topics = random.sample(CREATIVE_TOPICS, min(BATCH_SIZE, len(CREATIVE_TOPICS)))
        styles = random.choices(STYLES, k=BATCH_SIZE)
        lengths = random.choices(["200", "300", "400", "500", "600"], k=BATCH_SIZE)

        prompts = []
        for i in range(BATCH_SIZE):
            prompts.append(
                f"Write a {styles[i]} short story ({lengths[i]} words) about {topics[i]}. "
                f"Write naturally with vivid details and varied sentence structure."
            )

        numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(prompts))
        try:
            response = call_gemini(
                f"Complete each of these {BATCH_SIZE} creative writing tasks independently. "
                f"Separate each piece with '---'.\n\n{numbered}\n\n"
                f"Write each story naturally and independently. Separate with '---'."
            )

            parts = [p.strip() for p in response.split("---") if len(p.strip()) > 100]
            # Strip any numbering artifacts
            for part in parts:
                clean = part.strip()
                for prefix_pattern in ["[1]", "[2]", "[3]", "[4]", "[5]",
                                       "1.", "2.", "3.", "4.", "5."]:
                    if clean.startswith(prefix_pattern):
                        clean = clean[len(prefix_pattern):].strip()
                        break
                # Remove title lines that Gemini sometimes adds
                lines = clean.split("\n")
                if len(lines) > 2 and len(lines[0]) < 80 and lines[0] == lines[0].strip():
                    if lines[1].strip() == "":
                        clean = "\n".join(lines[2:]).strip()

                if len(clean) > 100:
                    texts.append(clean)
                    with open(progress_path, "a") as f:
                        f.write(json.dumps({"text": clean}) + "\n")

            if batch_idx % 20 == 0:
                print(f"  {len(texts)}/{n} generated (batch {batch_idx+1}/{batches_needed})")

            time.sleep(2)

        except Exception as e:
            print(f"  Error on batch {batch_idx}: {e}")
            time.sleep(10)
            continue

    texts = texts[:n]
    print(f"  Done: {len(texts)} AI creative samples")
    return texts


# ============================================================
# Step 3: Combine and tokenize
# ============================================================
def build_dataset():
    # Get human creative writing
    human_texts = get_writing_prompts()

    # Get AI creative writing
    ai_texts = generate_ai_creative()

    if not ai_texts:
        print("\nWARNING: No AI texts generated. Proceeding with human only.")

    # Build dataframe
    rows = []
    for text in human_texts:
        rows.append({"text": text, "label": 0, "source": "writingprompts", "domain": "creative"})
    for text in ai_texts:
        rows.append({"text": text, "label": 1, "source": "gemini_creative", "domain": "creative"})

    creative_df = pd.DataFrame(rows)
    print(f"\nCreative dataset: {len(creative_df)} samples")
    print(f"  Human: {(creative_df['label']==0).sum()}")
    print(f"  AI: {(creative_df['label']==1).sum()}")

    # Save raw
    creative_df.to_parquet(os.path.join(DATA_DIR, "creative_paired.parquet"), index=False)

    # Tokenize
    print("\nTokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        creative_df["text"].tolist(),
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
    )

    tokenized = pd.DataFrame({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "label": creative_df["label"].tolist(),
        "text": creative_df["text"].tolist(),
        "source": creative_df["source"].tolist(),
        "domain": creative_df["domain"].tolist(),
    })

    # Merge with original training data
    print("\nMerging with original training pool...")
    original = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    print(f"  Original train: {len(original)} samples")
    print(f"  Original columns: {list(original.columns)}")

    # Sample down original to keep pool size manageable (~60k total)
    original_budget = 50000
    if len(original) > original_budget:
        original = original.sample(n=original_budget, random_state=42)

    # Keep only the columns we need (text is there for FP mining)
    keep_cols = ["input_ids", "attention_mask", "label", "text", "source", "domain"]
    original = original[[c for c in keep_cols if c in original.columns]]
    tokenized = tokenized[[c for c in keep_cols if c in tokenized.columns]]

    active = pd.concat([original, tokenized], ignore_index=True)
    active_path = os.path.join(DATA_DIR, "active_train_local.parquet")
    active.to_parquet(active_path, index=False)

    print(f"\nFinal active training pool: {len(active)} samples")
    print(f"  Domain distribution:")
    print(active["domain"].value_counts().to_string())
    print(f"\n  Label balance: human={int((active['label']==0).sum())}, ai={int((active['label']==1).sum())}")
    print(f"  Saved to {active_path}")
    print(f"  File size: {os.path.getsize(active_path) / 1024 / 1024:.1f} MB")

    return active_path


if __name__ == "__main__":
    build_dataset()
