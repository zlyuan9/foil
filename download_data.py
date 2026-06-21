"""
Download and combine HC3, TuringBench, and GPABenchmark
into a unified dataset with columns: [text, label, source, domain]
label: 0 = human, 1 = ai
"""

import json
import os
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_rows = []

# --- HC3 (Human ChatGPT Comparison) ---
print("Downloading HC3...")
try:
    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    with open(path, "r") as f:
        for line in f:
            example = json.loads(line)
            question = example.get("question", "")
            for ans in example.get("human_answers", []):
                all_rows.append({
                    "text": f"{question}\n{ans}" if question else ans,
                    "label": 0,
                    "source": "hc3",
                    "domain": "qa",
                })
            for ans in example.get("chatgpt_answers", []):
                all_rows.append({
                    "text": f"{question}\n{ans}" if question else ans,
                    "label": 1,
                    "source": "hc3",
                    "domain": "qa",
                })
    print(f"  HC3: {len(all_rows)} samples")
except Exception as e:
    print(f"  HC3 failed: {e}")

# --- TuringBench ---
print("Downloading TuringBench...")
count_before = len(all_rows)
try:
    import csv
    import io

    path = hf_hub_download("turingbench/TuringBench", "TuringBench.zip", repo_type="dataset")
    with zipfile.ZipFile(path, "r") as z:
        csv_files = [n for n in z.namelist()
                     if n.endswith(".csv") and not n.startswith("__MACOSX")]
        for csv_name in csv_files:
            with z.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                header = next(reader)
                for row in reader:
                    if len(row) < 2:
                        continue
                    text = row[0].strip()
                    generator = row[1].strip()
                    if len(text) < 50:
                        continue
                    label = 0 if generator == "human" else 1
                    all_rows.append({
                        "text": text,
                        "label": label,
                        "source": "turingbench",
                        "domain": "news",
                        "generator": generator,
                    })
    print(f"  TuringBench: {len(all_rows) - count_before} samples added")
except Exception as e:
    print(f"  TuringBench failed: {e}")

# --- GPABenchmark ---
print("Downloading GPABenchmark...")
count_before = len(all_rows)
try:
    repo_files = list_repo_files("julianzy/GPABenchmark", repo_type="dataset")
    json_files = [f for f in repo_files if f.endswith(".json")]

    for fname in json_files:
        path = hf_hub_download("julianzy/GPABenchmark", fname, repo_type="dataset")
        # filename like GPABenchmark/CS_TASK1/gpt.json or .../hum.json
        is_ai = "gpt" in os.path.basename(fname)
        label = 1 if is_ai else 0
        # domain from path
        parts = fname.split("/")
        domain = parts[1] if len(parts) > 1 else "unknown"  # CS_TASK1, HSS_TASK2, etc.

        with open(path, "r") as f:
            data = json.load(f)

        # Handle both list of strings and list of dicts
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text", "") or item.get("essay", "") or item.get("content", "")
                else:
                    continue
                if len(text.strip()) > 50:
                    all_rows.append({
                        "text": text.strip(),
                        "label": label,
                        "source": "gpabench",
                        "domain": f"student_essays_{domain}",
                    })
        elif isinstance(data, dict):
            for key, val in data.items():
                text = val if isinstance(val, str) else str(val)
                if len(text.strip()) > 50:
                    all_rows.append({
                        "text": text.strip(),
                        "label": label,
                        "source": "gpabench",
                        "domain": f"student_essays_{domain}",
                    })
    print(f"  GPABenchmark: {len(all_rows) - count_before} samples added")
except Exception as e:
    print(f"  GPABenchmark failed: {e}")

# --- Combine and save ---
df = pd.DataFrame(all_rows)
df = df[df["text"].str.strip().str.len() > 0]

print(f"\n{'='*50}")
print(f"Combined Dataset")
print(f"{'='*50}")
print(f"Total samples: {len(df)}")
print(f"Human (0): {(df['label'] == 0).sum()}")
print(f"AI (1): {(df['label'] == 1).sum()}")
print(f"\nBy source:")
print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
print(f"\nBy domain:")
print(df.groupby(["domain", "label"]).size().unstack(fill_value=0))

output_path = os.path.join(OUTPUT_DIR, "combined_train.parquet")
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
