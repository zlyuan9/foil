"""
PyTorch Dataset and DataLoader for the preprocessed train/val/test splits.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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


def get_dataloaders(batch_size=32, num_workers=4, data_dir="data"):
    train_ds = TextDetectionDataset(f"{data_dir}/train.parquet")
    val_ds = TextDetectionDataset(f"{data_dir}/val.parquet")
    test_ds = TextDetectionDataset(f"{data_dir}/test.parquet")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    print(f"Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"Val:   {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    print(f"Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  labels: {batch['labels']}")
