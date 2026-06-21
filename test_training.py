"""
Test that model training works end-to-end on a tiny batch.
Validates: model init, forward, backward, optimizer step.
Mirrors the exact setup used in modal_train.py.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

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


def test_forward_backward():
    """Test model can do forward + backward pass (mimics Modal setup)."""
    print("TEST: Forward + backward pass...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Mimic modal_train.py: keep backbone native (fp16), classifier in fp32
    model = AITextDetector()
    model.classifier = model.classifier.float()
    model = model.to(device)

    # Dummy batch (batch_size=4, seq_len=128 to be fast)
    input_ids = torch.randint(0, 1000, (4, 128)).to(device)
    attention_mask = torch.ones(4, 128, dtype=torch.long).to(device)
    labels = torch.tensor([0, 1, 0, 1]).to(device)

    # Forward with autocast
    if device.type == "mps":
        output = model(input_ids, attention_mask, labels)
    else:
        with torch.amp.autocast("cuda"):
            output = model(input_ids, attention_mask, labels)

    loss = output["loss"]
    logits = output["logits"]

    assert loss is not None
    assert logits.shape == (4, 2)
    print(f"  Forward OK: loss={loss.item():.4f}, logits shape={logits.shape}")

    # Backward
    loss.backward()
    print("  Backward OK")

    # Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.step()
    optimizer.zero_grad()
    print("  Optimizer step OK")

    print("  PASS")


def test_memory_estimate():
    """Estimate memory usage for batch_size=16, seq_len=512."""
    print("\nTEST: Memory estimate for batch_size=16, seq_len=512...")

    model = AITextDetector()
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"  Model params memory: {param_memory_mb:.0f} MB")

    # Rough estimate: activations for DeBERTa ~ 4x param memory for batch_size=16, seq=512
    est_total = param_memory_mb * 4
    print(f"  Estimated total (params + grads + activations): ~{est_total:.0f} MB")
    print(f"  A10G has 24 GB -> {'SHOULD FIT' if est_total < 20000 else 'MIGHT OOM'}")

    # With fp16 backbone
    fp16_params = sum(p.numel() * 2 for p in model.backbone.parameters()) / 1024 / 1024
    fp32_params = sum(p.numel() * 4 for p in model.classifier.parameters()) / 1024 / 1024
    total_param_mem = fp16_params + fp32_params
    print(f"\n  With fp16 backbone + fp32 head:")
    print(f"    Backbone: {fp16_params:.0f} MB (fp16)")
    print(f"    Head: {fp32_params:.0f} MB (fp32)")
    print(f"    Total params: {total_param_mem:.0f} MB")
    est_total_mixed = total_param_mem * 3.5
    print(f"    Estimated total: ~{est_total_mixed:.0f} MB")
    print(f"    A10G has 24 GB -> {'SHOULD FIT' if est_total_mixed < 20000 else 'MIGHT OOM'}")

    print("  PASS")


if __name__ == "__main__":
    test_forward_backward()
    test_memory_estimate()
