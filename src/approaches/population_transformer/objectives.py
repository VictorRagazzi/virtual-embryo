"""Differentiable population objectives used by the E020 Transformer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def hybrid_gene_selection(variance: np.ndarray, temporal_change: np.ndarray, n_genes: int) -> np.ndarray:
    """Select half by train variance and half by train-only absolute temporal change."""
    variance = np.asarray(variance).ravel()
    temporal_change = np.asarray(temporal_change).ravel()
    if variance.shape != temporal_change.shape or not 0 < n_genes <= len(variance):
        raise ValueError("Invalid gene-selection inputs.")
    orders = (np.argsort(-variance, kind="stable"), np.argsort(-np.abs(temporal_change), kind="stable"))
    wanted = (n_genes // 2, n_genes - n_genes // 2)
    selected: list[int] = []
    total = 0
    for order, count in zip(orders, wanted, strict=True):
        total += count
        for index in order:
            if int(index) not in selected:
                selected.append(int(index))
                if len(selected) == total:
                    break
    for pair in zip(*orders):
        for index in pair:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == n_genes:
                return np.asarray(selected, dtype=np.int64)
    return np.asarray(selected[:n_genes], dtype=np.int64)


def soft_rank(values: torch.Tensor, temperature: float, block_size: int) -> torch.Tensor:
    """Pairwise sigmoid rank, O(G²) compute but O(G*block_size) temporary memory."""
    if temperature <= 0 or block_size <= 0:
        raise ValueError("temperature and block_size must be positive")
    flat = values.reshape(-1)
    chunks = []
    for start in range(0, flat.numel(), block_size):
        current = flat[start : start + block_size]
        chunks.append(1.0 + torch.sigmoid((flat[:, None] - current[None, :]) / temperature).sum(0))
    return torch.cat(chunks).reshape(values.shape)


def de_ranking_loss(predicted: torch.Tensor, target: torch.Tensor, reference: torch.Tensor,
                    *, temperature: float, top_k: int, block_size: int) -> torch.Tensor:
    """Smooth proxy for signed DE recovery and rank agreement on pseudobulk shifts.

    Rows are cells and columns are the train-selected genes. A sigmoid around the
    target's kth-largest absolute shift supplies a soft DE mask; pairwise sigmoid
    ranks replace scipy ranks/argsort, which are discrete in the real scorer.
    """
    dp = predicted.mean(0) - reference.mean(0)
    dt = target.mean(0) - reference.mean(0)
    k = min(max(int(top_k), 1), dt.numel())
    threshold = torch.topk(dt.detach().abs(), k).values[-1]
    mask = torch.sigmoid((dt.detach().abs() - threshold) / temperature)
    direction = (mask * torch.nn.functional.smooth_l1_loss(dp, dt, reduction="none")).sum() / mask.sum().clamp_min(1)
    rp = soft_rank(dp, temperature, block_size)
    rt = soft_rank(dt.detach(), temperature, block_size)
    rp = (rp - rp.mean()) / rp.std(unbiased=False).clamp_min(1e-6)
    rt = (rt - rt.mean()) / rt.std(unbiased=False).clamp_min(1e-6)
    return direction + (mask * (rp - rt).square()).sum() / mask.sum().clamp_min(1)


def multicut_signed_de_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    cuts: tuple[int, ...] | list[int],
    temperature: float = 0.1,
) -> torch.Tensor:
    """Recover positive and negative DE genes at multiple fixed cutoffs."""
    predicted = predicted_delta.reshape(-1)
    target = target_delta.detach().reshape(-1)
    if predicted.shape != target.shape or predicted.numel() < 2:
        raise ValueError("predicted_delta and target_delta must have the same gene dimension")
    if temperature <= 0 or not cuts:
        raise ValueError("temperature and at least one cutoff are required")
    losses = []
    genes = predicted.numel()
    for requested in cuts:
        k = min(max(int(requested), 1), genes - 1)
        positive = torch.zeros_like(target); negative = torch.zeros_like(target)
        positive[torch.topk(target, k).indices] = 1
        negative[torch.topk(-target, k).indices] = 1
        weight = torch.as_tensor((genes - k) / k, dtype=predicted.dtype, device=predicted.device)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
            predicted / temperature, positive, pos_weight=weight))
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
            -predicted / temperature, negative, pos_weight=weight))
    return torch.stack(losses).mean()


class RFFMMD(nn.Module):
    """Seeded random Fourier approximation of Gaussian-kernel MMD²."""

    def __init__(self, genes: int, features: int, sigma: float, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("omega", torch.randn(genes, features, generator=generator) / sigma)
        self.register_buffer("phase", 2 * torch.pi * torch.rand(features, generator=generator))
        self.scale = (2.0 / features) ** 0.5

    def embed(self, expression: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(expression @ self.omega + self.phase)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.embed(predicted).mean(0) - self.embed(target).mean(0)).square().sum()


def differentiable_population(centers: torch.Tensor, dispersions: torch.Tensor, groups: torch.Tensor,
                              residuals: torch.Tensor, decoder_weight: torch.Tensor,
                              decoder_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Generate expression from fixed residual draws and a frozen selected-gene decoder."""
    latent = centers[groups] + dispersions[groups] * residuals
    decoded = torch.nn.functional.linear(latent, decoder_weight, decoder_bias)
    return torch.nn.functional.softplus(decoded)


def relative_gain(model_error: float, copy_error: float, epsilon: float = 1e-8) -> float:
    return (copy_error - model_error) / max(copy_error, epsilon)


def selection_score(model: dict[str, float], copy_last: dict[str, float], epsilon: float = 1e-8) -> dict[str, float]:
    gains = {name: relative_gain(model[name], copy_last[name], epsilon) for name in
             ("de_ranking", "rff_mmd", "structure", "proportion")}
    gains["score"] = .4 * gains["de_ranking"] + .3 * gains["rff_mmd"] + .2 * gains["structure"] + .1 * gains["proportion"]
    return gains


@dataclass
class EarlyStopping:
    patience: int
    best: float = -float("inf")
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if score > self.best:
            self.best, self.bad_epochs = score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def stopped(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"patience": self.patience, "best": self.best, "bad_epochs": self.bad_epochs}

    @classmethod
    def from_state_dict(cls, state: dict[str, float | int]) -> "EarlyStopping":
        return cls(int(state["patience"]), float(state["best"]), int(state["bad_epochs"]))
