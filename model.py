"""
AI text detection model: DeBERTa-v3-base backbone + classification head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class AITextDetector(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for base

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    print("Loading model...")
    model = AITextDetector()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone hidden size: {model.backbone.config.hidden_size}")

    # Quick forward pass test
    model = model.float()
    dummy_input = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    dummy_labels = torch.tensor([0, 1])

    output = model(dummy_input, dummy_mask, dummy_labels)
    print(f"\nTest forward pass:")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
