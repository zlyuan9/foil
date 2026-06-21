"""
Evaluate saved model on full test set. Minimal GPU time.
"""

import modal

app = modal.App("pangram-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
)

volume = modal.Volume.from_name("pangram-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    gpu="A10G",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def evaluate():
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

    device = torch.device("cuda")
    MODEL_NAME = "microsoft/deberta-v3-base"

    class AITextDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME)
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

    # Load model
    model_path = f"{VOLUME_PATH}/best_model.pt"
    print(f"Loading model from {model_path}...")
    model = AITextDetector().float().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model.eval()
    print("  Model loaded.")

    # Load test set
    test_path = f"{VOLUME_PATH}/test.parquet"
    print(f"Loading test set from {test_path}...")
    test_ds = TextDetectionDataset(test_path)
    print(f"  {len(test_ds)} test samples")

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    # Evaluate
    all_logits, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
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

    # Metrics
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    auroc = roc_auc_score(labels_np, probs)
    avg_loss = total_loss / len(test_loader)

    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Loss:     {avg_loss:.4f}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1:       {f1:.4f}")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels_np, preds, target_names=["Human", "AI"]))
    print(f"Confusion Matrix:")
    cm = confusion_matrix(labels_np, preds)
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "confusion_matrix": cm.tolist(),
    }


@app.local_entrypoint()
def main():
    import subprocess

    # Make sure test.parquet is on volume
    result = subprocess.run(
        ["modal", "volume", "put", "pangram-data", "data/test.parquet", "/test.parquet"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Uploaded test.parquet")
    else:
        print(f"test.parquet: {result.stderr.strip() or 'already exists'}")

    print("Running evaluation...")
    result = evaluate.remote()
    print(f"\nResults: {result}")
