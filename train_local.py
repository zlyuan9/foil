"""
Local training pipeline for M4 Pro Mac (MPS backend).
Fixes from modal_train.py:
- FP scanning covers FULL training pool (not just 10k)
- Incorporates creative writing data from download_creative.py + generate_ai_creative.py
- MPS-compatible (no CUDA-specific ops)
- fp32 training (MPS fp16 support is unreliable)

Estimated time on M4 Pro 24GB: 8-12 hours for 5 iterations
"""

import os

# Must be set before torch import — allows MPS to spill into system RAM
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import time
import json
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# Config
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-base"
DATA_DIR = "data"
OUTPUT_DIR = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL_POOL_SIZE = 60000  # larger pool to include creative data
MAX_ITERATIONS = 3  # reduced for overnight feasibility on MPS
NUM_EPOCHS_PER_ITER = 2  # 2 epochs enough with good data
BATCH_SIZE = 2  # MPS needs tiny batches for DeBERTa
GRAD_ACCUM_STEPS = 16  # effective batch = 32
LEARNING_RATE = 2e-5
MAX_MIRRORS_PER_ITER = 1000  # 2x the original — actually generate enough
CONVERGENCE_THRESHOLD = 0.001
FP_SCAN_BATCH_SIZE = 16  # inference is much cheaper than training
VAL_EVAL_SIZE = 5000
MAX_SEQ_LENGTH = 256  # balance between coverage and MPS memory

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Using CPU (this will be slow)")


# ============================================================
# Model
# ============================================================
class AITextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        self.backbone.gradient_checkpointing_enable()
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================================
# Dataset
# ============================================================
class TextDetectionDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()
        self.labels = df["label"].tolist()
        self.texts = df["text"].tolist() if "text" in df.columns else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_text(self, idx):
        if self.texts:
            return self.texts[idx]
        return None


def collate_fn(batch):
    max_len = min(max(len(item["input_ids"]) for item in batch), MAX_SEQ_LENGTH)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = min(len(item["input_ids"]), max_len)
        input_ids[i, :seq_len] = item["input_ids"][:seq_len]
        attention_mask[i, :seq_len] = item["attention_mask"][:seq_len]
        labels[i] = item["label"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, labels)
        loss = output["loss"] / GRAD_ACCUM_STEPS
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if device.type == "mps":
                torch.mps.empty_cache()

    # Handle remaining gradients
    if (step + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids, attention_mask, labels)
        total_loss += output["loss"].item()
        all_logits.append(output["logits"].cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(labels_np, preds),
        "f1": f1_score(labels_np, preds),
        "auroc": roc_auc_score(labels_np, probs),
    }


def get_false_positives(model, dataset, batch_size=FP_SCAN_BATCH_SIZE):
    """
    Scan the FULL dataset for false positives.
    Fixed from original which only scanned 10k/50k samples.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,  # MPS doesn't love multiprocessing
    )
    model.eval()
    fp_indices = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Scanning for FPs", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            preds = model(input_ids, attention_mask)["logits"].argmax(dim=1).cpu()
            for i in range(len(labels)):
                if labels[i] == 0 and preds[i] == 1:
                    fp_indices.append(offset + i)
            offset += len(labels)

    return fp_indices


# ============================================================
# Mirror generation (Gemini)
# ============================================================
def generate_mirrors(fp_texts: list[str], max_mirrors: int = MAX_MIRRORS_PER_ITER) -> list[dict]:
    """Generate mirror examples from false positives using Gemini Flash."""
    from google import genai
    from google.genai import errors as genai_errors

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  WARNING: No GEMINI_API_KEY set, skipping mirror generation")
        return []

    client = genai.Client(api_key=api_key)
    GEMINI_MODEL = "gemini-2.5-flash"
    MIRROR_BATCH = 10

    def call_gemini(contents, retries=3):
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents
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

    if len(fp_texts) > max_mirrors:
        fp_texts = random.sample(fp_texts, max_mirrors)

    mirrors = []
    for i in range(0, len(fp_texts), MIRROR_BATCH):
        batch = fp_texts[i: i + MIRROR_BATCH]
        try:
            # Reverse-engineer prompts
            numbered = "\n\n".join(f"[{j+1}]\n\"\"\"{t[:1000]}\"\"\"" for j, t in enumerate(batch))
            prompt_response = call_gemini(
                f"For each of the following {len(batch)} texts, write a short prompt (1-2 sentences) "
                f"that someone might use to ask an AI to generate something very similar.\n\n{numbered}\n\n"
                f"Respond with ONLY the prompts, numbered [1] through [{len(batch)}]."
            )
            lines = [l.strip() for l in prompt_response.strip().split("\n") if l.strip()]
            prompts = []
            for line in lines:
                for prefix in [f"[{len(prompts)+1}]", f"{len(prompts)+1}.", f"{len(prompts)+1})"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if line:
                    prompts.append(line)
            while len(prompts) < len(batch):
                prompts.append(f"Write a text similar to: {batch[len(prompts)][:200]}")
            prompts = prompts[:len(batch)]

            time.sleep(1.5)

            # Generate mirrors
            numbered_prompts = "\n\n".join(f"[{j+1}] {p}" for j, p in enumerate(prompts))
            mirror_response = call_gemini(
                f"Write {len(prompts)} separate texts based on these prompts. "
                f"Each 100-400 words. Separate with '---'.\n\n{numbered_prompts}"
            )
            parts = [p.strip() for p in mirror_response.split("---") if len(p.strip()) > 100]
            for text in parts:
                mirrors.append({"text": text, "label": 1, "source": "mirror", "domain": "mirror"})

            time.sleep(1.5)

            if (i // MIRROR_BATCH) % 5 == 0:
                print(f"    Mirrors: {len(mirrors)}/{max_mirrors}")

        except Exception as e:
            print(f"    Mirror error at batch {i}: {e}")
            time.sleep(5)
            continue

    return mirrors


# ============================================================
# Data preparation
# ============================================================
def prepare_training_data(tokenizer):
    """
    Combine original data with creative augmentation data.
    Returns path to the tokenized active training pool.
    """
    active_path = os.path.join(DATA_DIR, "active_train_local.parquet")

    if os.path.exists(active_path):
        print(f"Active pool already exists at {active_path}")
        df = pd.read_parquet(active_path)
        print(f"  {len(df)} samples")
        return active_path

    print("Building active training pool...")

    # Load original training data
    original = pd.read_parquet(os.path.join(DATA_DIR, "combined_train.parquet"))
    print(f"  Original data: {len(original)} samples")

    # Sample from original (keep it manageable)
    original_sample = original.sample(n=min(40000, len(original)), random_state=42)

    # Load creative human data if available
    creative_human_path = os.path.join(DATA_DIR, "creative_human.parquet")
    creative_ai_path = os.path.join(DATA_DIR, "creative_ai.parquet")

    parts = [original_sample]

    if os.path.exists(creative_human_path):
        creative_human = pd.read_parquet(creative_human_path)
        print(f"  Creative human data: {len(creative_human)} samples")
        parts.append(creative_human)
    else:
        print("  WARNING: No creative_human.parquet found. Run download_creative.py first!")

    if os.path.exists(creative_ai_path):
        creative_ai = pd.read_parquet(creative_ai_path)
        print(f"  Creative AI data: {len(creative_ai)} samples")
        parts.append(creative_ai)
    else:
        print("  WARNING: No creative_ai.parquet found. Run generate_ai_creative.py first!")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined[combined["text"].str.strip().str.len() > 100]
    print(f"  Combined pool: {len(combined)} samples")

    # Cap at INITIAL_POOL_SIZE with stratified sampling
    if len(combined) > INITIAL_POOL_SIZE:
        # Keep all creative data, sample from original
        creative_mask = combined["domain"].isin(["creative", "reviews", "online_content", "encyclopedic", "books"])
        creative_part = combined[creative_mask]
        original_part = combined[~creative_mask]

        remaining_budget = INITIAL_POOL_SIZE - len(creative_part)
        if remaining_budget > 0 and len(original_part) > remaining_budget:
            original_part = original_part.sample(n=remaining_budget, random_state=42)
        combined = pd.concat([creative_part, original_part], ignore_index=True)

    print(f"  Final pool size: {len(combined)}")
    print(f"  Domain distribution:")
    print(combined["domain"].value_counts().to_string())

    # Tokenize
    print("  Tokenizing...")
    encoded = tokenizer(
        combined["text"].tolist(),
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
    )

    tokenized = pd.DataFrame({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "label": combined["label"].tolist(),
        "text": combined["text"].tolist(),
        "source": combined["source"].tolist(),
        "domain": combined["domain"].tolist(),
    })

    tokenized.to_parquet(active_path, index=False)
    print(f"  Saved to {active_path}")
    return active_path


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("=" * 60)
    print("PANGRAM LOCAL TRAINING (M4 Pro MPS)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    active_path = prepare_training_data(tokenizer)
    val_path = os.path.join(DATA_DIR, "val.parquet")
    log_path = os.path.join(OUTPUT_DIR, "training_log_local.json")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    # Resume logic
    start_iteration = 0
    training_log = []
    prev_val_loss = float("inf")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            training_log = json.load(f)
        if training_log:
            start_iteration = len(training_log)
            prev_val_loss = training_log[-1]["val_loss"]
            print(f"\nRESUMING from iteration {start_iteration + 1}")

    for iteration in range(start_iteration, MAX_ITERATIONS):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{MAX_ITERATIONS}")
        print("=" * 60)

        # Load data
        train_ds = TextDetectionDataset(active_path)
        val_ds = TextDetectionDataset(val_path)

        val_subset = torch.utils.data.Subset(
            val_ds, list(range(min(VAL_EVAL_SIZE, len(val_ds))))
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # Initialize model
        model = AITextDetector().float().to(device)
        if iteration > 0 and os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(best_model_path, weights_only=False, map_location=device)
            )
            print("  Loaded previous best model")

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * NUM_EPOCHS_PER_ITER
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Train
        for epoch in range(NUM_EPOCHS_PER_ITER):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            val_metrics = evaluate(model, val_loader)
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS_PER_ITER}: "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"acc={val_metrics['accuracy']:.4f} "
                  f"f1={val_metrics['f1']:.4f} "
                  f"auroc={val_metrics['auroc']:.4f}")

        # Save checkpoint
        val_loss = val_metrics["loss"]
        iter_model_path = os.path.join(OUTPUT_DIR, f"model_iter{iteration+1}.pt")
        torch.save(model.state_dict(), iter_model_path)

        if val_loss < prev_val_loss:
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ New best model (val_loss={val_loss:.4f})")

        # Log
        iter_time = time.time() - iter_start
        training_log.append({
            "iteration": iteration + 1,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "pool_size": len(train_ds),
            "time_minutes": iter_time / 60,
        })
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"  Iteration time: {iter_time/60:.1f} min")

        # Convergence check
        delta = prev_val_loss - val_loss
        if abs(delta) < CONVERGENCE_THRESHOLD and iteration > 0:
            print(f"  CONVERGED (delta={abs(delta):.6f})")
            break
        prev_val_loss = val_loss

        # ============================================================
        # Hard negative mining — FULL POOL SCAN
        # ============================================================
        print(f"\n  Scanning FULL pool ({len(train_ds)} samples) for false positives...")
        fp_indices = get_false_positives(model, train_ds)
        print(f"  Found {len(fp_indices)} false positives")

        if not fp_indices:
            print("  No FPs found — model is well-calibrated on training data")
            continue

        # Extract FP texts
        fp_texts = [train_ds.get_text(i) for i in fp_indices if train_ds.get_text(i)]
        fp_texts = [t for t in fp_texts if t]
        print(f"  Got {len(fp_texts)} FP texts for mirror generation")

        if not fp_texts:
            print("  No FP texts available, skipping mirror generation")
            continue

        # Generate mirrors
        print(f"  Generating mirrors (max {MAX_MIRRORS_PER_ITER})...")
        mirrors = generate_mirrors(fp_texts, max_mirrors=MAX_MIRRORS_PER_ITER)
        print(f"  Generated {len(mirrors)} mirrors")

        if mirrors:
            # Tokenize and add to pool
            mirror_texts = [m["text"] for m in mirrors]
            encoded = tokenizer(
                mirror_texts, truncation=True, max_length=512,
                padding=False, return_attention_mask=True,
            )
            mirror_df = pd.DataFrame({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "label": [m["label"] for m in mirrors],
                "text": mirror_texts,
                "source": [m["source"] for m in mirrors],
                "domain": [m["domain"] for m in mirrors],
            })
            active_df = pd.read_parquet(active_path)
            active_df = pd.concat([active_df, mirror_df], ignore_index=True)
            active_df.to_parquet(active_path, index=False)
            print(f"  Pool updated: {len(active_df)} samples (+{len(mirrors)} mirrors)")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_model_path}")
    print(f"Log: {log_path}")

    # Final summary
    if training_log:
        best = min(training_log, key=lambda x: x["val_loss"])
        print(f"\nBest iteration: {best['iteration']}")
        print(f"  Val loss:     {best['val_loss']:.4f}")
        print(f"  Val accuracy: {best['val_accuracy']:.4f}")
        print(f"  Val F1:       {best['val_f1']:.4f}")
        print(f"  Val AUROC:    {best['val_auroc']:.4f}")


if __name__ == "__main__":
    main()
