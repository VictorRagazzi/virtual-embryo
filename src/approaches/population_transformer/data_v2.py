"""Leakage-safe split and residual helpers for E020."""

from __future__ import annotations

import numpy as np


def official_split(labels: np.ndarray, n_holdout: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic approximately stratified holdout; labels are never model features."""
    labels = np.asarray(labels).astype(str)
    if not 0 < n_holdout < len(labels):
        raise ValueError("n_holdout must be between zero and population size")
    rng = np.random.default_rng(seed)
    groups, counts = np.unique(labels, return_counts=True)
    quotas = counts * n_holdout / len(labels)
    allocation = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - allocation), kind="stable")[: n_holdout - allocation.sum()]:
        allocation[index] += 1
    holdout = np.concatenate([rng.choice(np.flatnonzero(labels == group), size=count, replace=False)
                              for group, count in zip(groups, allocation, strict=True)])
    holdout = np.sort(holdout.astype(np.int64))
    train = np.setdiff1d(np.arange(len(labels), dtype=np.int64), holdout, assume_unique=True)
    return train, holdout


def residual_indices(requested_groups: np.ndarray, source_groups: np.ndarray, source_centers: np.ndarray,
                     predicted_centers: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """Fixed seeded draws, falling back to the nearest non-empty source group."""
    rng = np.random.default_rng(seed)
    available = np.unique(source_groups)
    chosen, fallback = [], 0
    for group in requested_groups:
        candidates = np.flatnonzero(source_groups == group)
        if not len(candidates):
            nearest = available[np.argmin(((source_centers[available] - predicted_centers[group]) ** 2).sum(1))]
            candidates, fallback = np.flatnonzero(source_groups == nearest), fallback + 1
        chosen.append(rng.choice(candidates))
    return np.asarray(chosen, dtype=np.int64), fallback
