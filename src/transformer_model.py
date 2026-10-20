"""Configuration-driven Transformer classifier used by protocol-v2 comparators."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from transformers import AutoModel


SUPPORTED_POOLING = {"pooler_output", "first_token"}


class TransformerSentimentClassifier(nn.Module):
    """A pinned backbone with the historical linear or explicit GELU MLP head."""

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        pooling: str,
        num_classes: int = 3,
        dropout: float = 0.3,
        local_files_only: bool = False,
        classifier_type: str = "linear",
        classifier_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if pooling not in SUPPORTED_POOLING:
            raise ValueError(
                f"Unsupported pooling strategy {pooling!r}; expected one of {sorted(SUPPORTED_POOLING)}"
            )

        self.model_id = model_id
        self.model_revision = model_revision
        self.pooling = pooling
        self.num_classes = num_classes
        self.backbone = AutoModel.from_pretrained(
            model_id,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier_type = classifier_type
        if classifier_type == "linear" and classifier_hidden_size is None:
            # Preserve the exact module names and computation of accepted checkpoints.
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        elif classifier_type == "mlp" and classifier_hidden_size in (128, 256):
            self.hidden_layer = nn.Linear(self.hidden_size, classifier_hidden_size)
            self.activation = nn.GELU()
            self.classifier = nn.Linear(classifier_hidden_size, num_classes)
        else:
            raise ValueError("Head must be linear or a GELU MLP with width 128/256")

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "pooler_output":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError(
                    "The configured pooler_output is unavailable from this backbone; "
                    "use a verified first_token configuration instead."
                )
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        if self.classifier_type == "mlp":
            return self.classifier(self.dropout(self.activation(self.hidden_layer(pooled))))
        return self.classifier(self.dropout(pooled))


def get_transformer_model(
    model_config: Mapping[str, Any],
    *,
    dropout: float,
    device,
    local_files_only: bool = False,
) -> TransformerSentimentClassifier:
    """Build a classifier exclusively from a validated model configuration."""

    head = model_config["project_head"]
    model = TransformerSentimentClassifier(
        model_id=model_config["model_id"],
        model_revision=model_config["model_revision"],
        pooling=head["pooling"],
        num_classes=int(head["num_classes"]),
        dropout=dropout,
        local_files_only=local_files_only,
        classifier_type=head["classifier"],
        classifier_hidden_size=head.get("hidden_size"),
    )
    return model.to(device)
