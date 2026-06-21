"""
Filter by length, truncate to 512, and create train/val/test splits.
Stratified by label and source to keep class/domain balance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
import os

DATA_PATH = "data/combined_tokenized.parquet"
OUTPUT_DIR = "data"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
MIN_LENGTH = 64
SEED = 42

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df)} samples loaded")

# Filter by length
before = len(df)
df = df[df["token_length"] >= MIN_LENGTH]
print(f"  Filtered < {MIN_LENGTH} tokens: {before} -> {len(df)} ({before - len(df)} removed)")

# Tokenize with truncation at 512
print(f"\nTokenizing with truncation at {MAX_LENGTH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_input_ids = []
all_attention_masks = []
texts = df["text"].tolist()

for i in tqdm(range(0, len(texts), 10000), desc="Tokenizing"):
    batch = texts[i : i + 10000]
    encoded = tokenizer(
        batch,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )
    all_input_ids.extend(encoded["input_ids"])
    all_attention_masks.extend(encoded["attention_mask"])

df["input_ids"] = all_input_ids
df["attention_mask"] = all_attention_masks

# Stratified split: 80% train, 10% val, 10% test
print("\nCreating stratified splits...")
df["stratify_key"] = df["label"].astype(str) + "_" + df["source"]

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["stratify_key"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["stratify_key"]
)

# Drop helper columns (keep text for mirror generation)
for split_df in [train_df, val_df, test_df]:
    split_df.drop(columns=["stratify_key", "token_length"], inplace=True, errors="ignore")

# Save
train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)

print(f"\n{'='*50}")
print("Split Summary")
print(f"{'='*50}")
print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")
print(f"\nLabel distribution:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    h = (split["label"] == 0).sum()
    a = (split["label"] == 1).sum()
    print(f"  {name}: human={h} ({h/len(split)*100:.1f}%), ai={a} ({a/len(split)*100:.1f}%)")

print(f"\nSaved to {OUTPUT_DIR}/{{train,val,test}}.parquet")
