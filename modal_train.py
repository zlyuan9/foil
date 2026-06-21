"""
Full training pipeline on Modal with A10G GPU.
Uploads data, runs hard negative mining loop, saves model to volume.
"""

import modal

app = modal.App("pangram-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "tqdm",
        "google-genai",
        "python-dotenv",
        "sentencepiece",
        "protobuf",
    )
)

volume = modal.Volume.from_name("pangram-data", create_if_missing=True)

VOLUME_PATH = "/data"
GEMINI_API_KEY = modal.Secret.from_name("gemini-api-key")


@app.function(
    gpu="A10G",
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[GEMINI_API_KEY],
    timeout=86400,  # 24 hours max
)
def train_pipeline(
    initial_sample_size: int = 50000,
    max_mirrors_per_iter: int = 500,
    convergence_threshold: float = 0.001,
    max_iterations: int = 10,
    batch_size: int = 8,
    num_epochs_per_iter: int = 3,
    learning_rate: float = 2e-5,
    grad_accum_steps: int = 4,  # effective batch size = 8 * 4 = 32
):
    import os
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
    from google import genai
    from google.genai import errors as genai_errors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # ============================================================
    # Model
    # ============================================================
    MODEL_NAME = "microsoft/deberta-v3-base"

    class AITextDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME)
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
            cls_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_output)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

    # ============================================================
    # Dataset / DataLoader
    # ============================================================
    class TextDetectionDataset(Dataset):
        def __init__(self, parquet_path):
            df = pd.read_parquet(parquet_path)
            self.input_ids = df["input_ids"].tolist()
            self.attention_mask = df["attention_mask"].tolist()
            self.labels = df["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.zeros(len(batch), dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = item["input_ids"]
            attention_mask[i, :seq_len] = item["attention_mask"]
            labels[i] = item["label"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # ============================================================
    # Training helpers
    # ============================================================
    def train_one_epoch(model, loader, optimizer, scheduler, scaler):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                output = model(input_ids, attention_mask, labels)
                loss = output["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_logits, all_labels = [], []
        total_loss = 0
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda"):
                output = model(input_ids, attention_mask, labels)
            total_loss += output["loss"].item()
            all_logits.append(output["logits"].float().cpu())
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

    def get_false_positive_indices(model, dataset, max_samples=10000):
        subset_size = min(max_samples, len(dataset))
        subset = torch.utils.data.Subset(dataset, list(range(subset_size)))
        loader = DataLoader(subset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4)
        model.eval()
        fp_indices = []
        offset = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Finding FPs"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                with torch.amp.autocast("cuda"):
                    preds = model(input_ids, attention_mask)["logits"].argmax(dim=1).cpu()
                for i in range(len(labels)):
                    if labels[i] == 0 and preds[i] == 1:
                        fp_indices.append(offset + i)
                offset += len(labels)
        return fp_indices

    # ============================================================
    # Mirror generation
    # ============================================================
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_MODEL = "gemini-2.5-flash"
    MIRROR_BATCH_SIZE = 10

    def call_gemini(contents, retries=3):
        for attempt in range(retries):
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents
                )
                if response.text is None:
                    raise ValueError("Empty response (safety filter)")
                return response.text
            except genai_errors.ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 2 ** (attempt + 1) * 5
                    print(f"    Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif "API key" in str(e):
                    raise
                else:
                    time.sleep(2 ** attempt)
            except genai_errors.ServerError:
                time.sleep(2 ** (attempt + 1) * 3)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini failed after {retries} retries")

    def generate_mirrors(texts, max_mirrors=500):
        if len(texts) > max_mirrors:
            texts = random.sample(texts, max_mirrors)

        mirrors = []
        for i in range(0, len(texts), MIRROR_BATCH_SIZE):
            batch = texts[i : i + MIRROR_BATCH_SIZE]
            try:
                # Step 1: reverse-engineer prompts
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

                time.sleep(1)

                # Step 2: generate mirror texts
                numbered_prompts = "\n\n".join(f"[{j+1}] {p}" for j, p in enumerate(prompts))
                mirror_response = call_gemini(
                    f"Write {len(prompts)} separate texts based on these prompts. "
                    f"Each 100-300 words. Separate with '---'.\n\n{numbered_prompts}"
                )
                parts = [p.strip() for p in mirror_response.split("---") if len(p.strip()) > 100]
                for text in parts:
                    mirrors.append({"text": text, "label": 1, "source": "mirror", "domain": "mirror"})

                time.sleep(1)
                print(f"  Mirror batch {i//MIRROR_BATCH_SIZE + 1}: {len(mirrors)} total")

            except Exception as e:
                print(f"  Mirror batch error at {i}: {e}")
                time.sleep(3)
                continue

        return mirrors

    # ============================================================
    # Main pipeline
    # ============================================================
    train_path = f"{VOLUME_PATH}/train.parquet"
    val_path = f"{VOLUME_PATH}/val.parquet"
    active_path = f"{VOLUME_PATH}/active_train.parquet"
    model_save_path = f"{VOLUME_PATH}/best_model.pt"
    log_path = f"{VOLUME_PATH}/training_log.json"

    # Quick Gemini connectivity test
    print("Testing Gemini API connection...")
    try:
        test_response = call_gemini("Say 'hello' in one word.")
        print(f"  Gemini OK: {test_response.strip()}")
    except Exception as e:
        print(f"  WARNING: Gemini test failed: {e}")
        print("  Mirror generation will likely fail. Continuing with training only...")

    # Resume logic: check if a previous run was interrupted
    print("\n" + "=" * 60)
    print("INITIALIZING PIPELINE")
    print("=" * 60)

    start_iteration = 0
    training_log = []
    prev_val_loss = float("inf")

    if os.path.exists(log_path) and os.path.exists(model_save_path) and os.path.exists(active_path):
        with open(log_path, "r") as f:
            training_log = json.load(f)
        if training_log:
            start_iteration = len(training_log)
            prev_val_loss = training_log[-1]["val_loss"]
            print(f"  RESUMING from iteration {start_iteration + 1} (prev val_loss={prev_val_loss:.4f})")
            print(f"  Loaded {len(training_log)} previous log entries")

    if start_iteration == 0:
        full_train_df = pd.read_parquet(train_path)
        print(f"Full training pool: {len(full_train_df)} samples")

        if len(full_train_df) > initial_sample_size:
            active_df = full_train_df.sample(n=initial_sample_size, random_state=42)
        else:
            active_df = full_train_df
        active_df.to_parquet(active_path, index=False)
        print(f"Initial active pool: {len(active_df)} samples")
    else:
        print(f"  Using existing active pool at {active_path}")

    import pyarrow.parquet as pq
    has_text = "text" in pq.read_schema(active_path).names
    print(f"  has_text: {has_text}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for iteration in range(start_iteration, max_iterations):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 60)

        # Train
        train_ds = TextDetectionDataset(active_path)
        val_ds = TextDetectionDataset(val_path)

        # Use subset of val for intermediate eval (full val is 143k — too slow)
        val_eval_size = min(10000, len(val_ds))
        val_subset = torch.utils.data.Subset(val_ds, list(range(val_eval_size)))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4,
        )
        val_loader = DataLoader(
            val_subset, batch_size=64, shuffle=False,
            collate_fn=collate_fn, num_workers=4,
        )

        model = AITextDetector().float().to(device)
        if iteration > 0 and os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, weights_only=False, map_location=device))
        scaler = torch.amp.GradScaler("cuda")

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs_per_iter
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        for epoch in range(num_epochs_per_iter):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_metrics = evaluate(model, val_loader)
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"acc={val_metrics['accuracy']:.4f} "
                  f"f1={val_metrics['f1']:.4f} "
                  f"auroc={val_metrics['auroc']:.4f}")

        # Save model (always save latest, and track best)
        torch.save(model.state_dict(), model_save_path)
        val_loss = val_metrics["loss"]
        best_model_path = f"{VOLUME_PATH}/best_model_by_val_loss.pt"
        if not os.path.exists(best_model_path) or val_loss < prev_val_loss:
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved (val_loss={val_loss:.4f})")
        volume.commit()

        # Check convergence
        delta = prev_val_loss - val_loss
        print(f"\n  Val loss delta: {delta:.4f}")

        training_log.append({
            "iteration": iteration + 1,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "delta": delta,
            "pool_size": len(train_ds),
        })
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        volume.commit()

        if abs(delta) < convergence_threshold and iteration > 0:
            print(f"  CONVERGED! Delta {abs(delta):.6f} < {convergence_threshold}")
            break

        prev_val_loss = val_loss

        # Find false positives
        print("\n  Finding false positives on training subset...")
        fp_indices = get_false_positive_indices(model, train_ds)
        print(f"  Found {len(fp_indices)} false positives")

        if not fp_indices:
            print("  No FPs — stopping.")
            break

        # Get FP texts for mirror generation
        active_df = pd.read_parquet(active_path)

        if "text" in active_df.columns:
            fp_texts = [active_df.iloc[i]["text"] for i in fp_indices if i < len(active_df)]
            print(f"  Extracted {len(fp_texts)} FP texts from active pool")
        else:
            # Fall back to val set
            print("  No text in active pool, trying val set...")
            val_df = pd.read_parquet(val_path)
            if "text" in val_df.columns:
                val_fp = get_false_positive_indices(model, val_ds)
                fp_texts = [val_df.iloc[i]["text"] for i in val_fp if i < len(val_df)]
                print(f"  Extracted {len(fp_texts)} FP texts from val set")
            else:
                print("  ERROR: No text column available anywhere. Stopping.")
                break

        if not fp_texts:
            print("  No FP texts available. Stopping.")
            break

        # Generate mirrors
        print(f"\n  Starting mirror generation from {len(fp_texts)} FPs (max {max_mirrors_per_iter})...")
        try:
            mirrors = generate_mirrors(fp_texts, max_mirrors=max_mirrors_per_iter)
            print(f"  Successfully generated {len(mirrors)} mirrors")
        except Exception as e:
            print(f"  ERROR in mirror generation: {e}")
            mirrors = []

        if mirrors:
            # Tokenize and add to pool
            print("  Tokenizing mirrors and adding to pool...")
            mirror_texts = [m["text"] for m in mirrors]
            encoded = tokenizer(
                mirror_texts, truncation=True, max_length=512,
                padding=False, return_attention_mask=True,
            )
            mirror_df = pd.DataFrame({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "label": [m["label"] for m in mirrors],
                "source": [m["source"] for m in mirrors],
                "domain": [m["domain"] for m in mirrors],
            })
            active_df = pd.read_parquet(active_path)
            active_df = pd.concat([active_df, mirror_df], ignore_index=True)
            active_df.to_parquet(active_path, index=False)
            volume.commit()
            print(f"  Pool updated: {len(active_df)} samples")
        else:
            print("  No mirrors generated this iteration, continuing anyway...")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final metrics: {val_metrics}")
    print(f"Training log saved to {log_path}")
    return training_log


@app.local_entrypoint()
def main():
    import subprocess

    # Upload data to volume (skip if already exists)
    print("Uploading data to Modal volume...")
    for fname in ["train.parquet", "val.parquet"]:
        result = subprocess.run(
            ["modal", "volume", "put", "pangram-data", f"data/{fname}", f"/{fname}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Uploaded {fname}")
        elif "already exists" in result.stderr:
            print(f"  {fname} already exists, skipping")
        else:
            print(f"  Warning: {result.stderr.strip()}")

    print("Starting training...")
    result = train_pipeline.remote()
    print(f"\nTraining complete!")
    print(f"Results: {result}")
