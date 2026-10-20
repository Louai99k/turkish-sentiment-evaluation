"""Numerically stable multiclass focal loss for the fixed three-class task."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MulticlassFocalLoss(nn.Module):
    """Mean of -alpha[y] * (1 - p[y])**gamma * log(p[y]).

    ``alpha=None`` means every class has unit weight. If alpha is supplied,
    its values are used directly; reduction is the arithmetic mean over
    examples, not PyTorch weighted-CE's sum-of-weights normalization.
    """

    def __init__(self, *, gamma: float, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Focal gamma must be finite and non-negative")
        self.gamma = float(gamma)
        if alpha is not None:
            if alpha.ndim != 1 or not bool(torch.isfinite(alpha).all()) or not bool((alpha > 0).all()):
                raise ValueError("Focal alpha must be a positive finite one-dimensional tensor")
            self.register_buffer("alpha", alpha.detach().clone())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("Focal loss expects [batch, classes] logits and [batch] labels")
        log_p_t = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        # -expm1(log_p_t) is stable when p_t is close to one.
        one_minus_p_t = -torch.expm1(log_p_t)
        per_example = -one_minus_p_t.pow(self.gamma) * log_p_t
        if self.alpha is not None:
            per_example = per_example * self.alpha[labels]
        return per_example.mean()
