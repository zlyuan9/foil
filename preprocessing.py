"""
Tokenize the combined dataset and report length statistics.
No truncation or filtering yet — just adds token lengths so we can decide cutoffs.
Processes in batches to avoid OOM.
"""

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_PATH = "data/combined_train.parquet"
OUTPUT_PATH = "data/combined_tokenized.parquet"
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 10000

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df)} samples loaded")

print("Tokenizing in batches...")
token_lengths = []
texts = df["text"].tolist()

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing"):
    batch = texts[i : i + BATCH_SIZE]
    encoded = tokenizer(
        batch,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )
    token_lengths.extend(encoded["length"])

df["token_length"] = token_lengths

print(f"\n{'='*50}")
print("Token Length Statistics")
print(f"{'='*50}")
print(df["token_length"].describe())
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = df["token_length"].quantile(p / 100)
    print(f"  {p:3d}th percentile: {val:.0f} tokens")

print(f"\nBy source:")
print(df.groupby("source")["token_length"].describe().round(0))

print(f"\nBy label (0=human, 1=ai):")
print(df.groupby("label")["token_length"].describe().round(0))

print(f"\nSamples > 512 tokens: {(df['token_length'] > 512).sum()} ({(df['token_length'] > 512).mean()*100:.1f}%)")
print(f"Samples > 1024 tokens: {(df['token_length'] > 1024).sum()} ({(df['token_length'] > 1024).mean()*100:.1f}%)")
print(f"Samples < 32 tokens: {(df['token_length'] < 32).sum()} ({(df['token_length'] < 32).mean()*100:.1f}%)")
print(f"Samples < 64 tokens: {(df['token_length'] < 64).sum()} ({(df['token_length'] < 64).mean()*100:.1f}%)")

df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved tokenized dataset to {OUTPUT_PATH}")
