"""Contratos e baselines temporais para tokens populacionais fixos."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


FIELDS = ("proportion", "latent_mean", "latent_dispersion")


def validate_token_state(state: dict[str, np.ndarray]) -> None:
    proportion, centers, dispersion = (state[name] for name in FIELDS)
    if proportion.ndim != 1 or centers.ndim != 2 or dispersion.shape != centers.shape:
        raise ValueError("Shapes inválidos para estado populacional.")
    if centers.shape[0] != len(proportion):
        raise ValueError("Número de grupos incompatível.")
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("Tokens contêm valores não finitos.")
    if (proportion < 0).any() or not np.isclose(proportion.sum(), 1.0, atol=1e-5):
        raise ValueError("Proporções inválidas.")
    if (dispersion < 0).any():
        raise ValueError("Dispersões negativas.")


def state_at(tokens: np.lib.npyio.NpzFile, index: int) -> dict[str, np.ndarray]:
    state = {name: np.asarray(tokens[name][index], dtype=np.float32).copy() for name in FIELDS}
    validate_token_state(state)
    return state


def state_for_source(cache: np.lib.npyio.NpzFile, tokens: np.lib.npyio.NpzFile, source: str) -> dict[str, np.ndarray]:
    """Resume um arquivo-fonte isoladamente, evitando fundir estágios homônimos."""
    mask = cache["sources"] == source
    if not mask.any():
        raise ValueError(f"Fonte ausente no cache: {source}")
    latent, labels = cache["latent"][mask], tokens["labels"][mask]
    k = tokens["centers"].shape[0]
    counts = np.bincount(labels, minlength=k)
    means = np.zeros((k, latent.shape[1]), dtype=np.float32)
    dispersion = np.zeros_like(means)
    for group in np.flatnonzero(counts):
        values = latent[labels == group]
        means[group] = values.mean(0)
        dispersion[group] = values.std(0)
    state = {"proportion": (counts / counts.sum()).astype(np.float32),
             "latent_mean": means, "latent_dispersion": dispersion}
    validate_token_state(state)
    return state


def copy_last(previous: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in previous.items()}


def linear_extrapolation(first: dict[str, np.ndarray], second: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    result = {name: second[name] + factor * (second[name] - first[name]) for name in FIELDS}
    logits = np.maximum(result["proportion"], 0)
    result["proportion"] = (logits / logits.sum() if logits.sum() else np.full_like(logits, 1 / len(logits))).astype(np.float32)
    result["latent_dispersion"] = np.maximum(result["latent_dispersion"], 0).astype(np.float32)
    result["latent_mean"] = result["latent_mean"].astype(np.float32)
    validate_token_state(result)
    return result


def token_metrics(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    observed = target["proportion"] > 0
    prop_mse = float(np.mean(np.square(prediction["proportion"] - target["proportion"])))
    center_mse = float(np.mean(np.square(prediction["latent_mean"][observed] - target["latent_mean"][observed])))
    dispersion_mse = float(np.mean(np.square(prediction["latent_dispersion"][observed] - target["latent_dispersion"][observed])))
    return {"proportion_mse": prop_mse, "center_mse_observed": center_mse,
            "dispersion_mse_observed": dispersion_mse, "aggregate_mse": prop_mse + center_mse + dispersion_mse,
            "proportion_l1": float(np.abs(prediction["proportion"] - target["proportion"]).sum()),
            "predicted_nonempty_groups": int((prediction["proportion"] > 1e-6).sum()),
            "target_nonempty_groups": int(observed.sum())}


def token_features(states: list[dict[str, np.ndarray]], times: list[float], target_time: float,
                   center_mean: float, center_scale: float, dispersion_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Cria sequência [tempo, proporção, centro, dispersão, vazio] para dois estágios."""
    chunks, groups = [], []
    for state, time in zip(states, times, strict=True):
        nonempty = state["proportion"] > 0
        centers = (state["latent_mean"] - center_mean) / center_scale
        dispersions = state["latent_dispersion"] / dispersion_scale
        centers[~nonempty] = 0
        dispersions[~nonempty] = 0
        time_columns = np.full((len(nonempty), 1), (time - target_time), dtype=np.float32)
        chunks.append(np.concatenate([time_columns, state["proportion"][:, None], centers, dispersions,
                                      (~nonempty)[:, None].astype(np.float32)], axis=1))
        groups.append(np.arange(len(nonempty), dtype=np.int64))
    return np.concatenate(chunks).astype(np.float32), np.concatenate(groups)


class PopulationTransformer(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, groups: int, model_dim: int = 128,
                 heads: int = 4, layers: int = 2, dropout: float = 0.05, program_dim: int = 0):
        super().__init__()
        self.groups = groups
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.group_embedding = nn.Embedding(groups, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 2, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.proportion_head = nn.Linear(model_dim, 1)
        self.center_head = nn.Linear(model_dim, latent_dim)
        self.dispersion_head = nn.Linear(model_dim, latent_dim)
        self.program_head = nn.Linear(model_dim, program_dim) if program_dim else None

    def forward(self, features: torch.Tensor, group_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(self.input_projection(features) + self.group_embedding(group_ids))[:, -self.groups:]
        result = {"proportion": torch.softmax(self.proportion_head(hidden).squeeze(-1), dim=-1),
                  "latent_mean": self.center_head(hidden),
                  "latent_dispersion": torch.nn.functional.softplus(self.dispersion_head(hidden))}
        if self.program_head is not None:
            result["program_delta"] = self.program_head(hidden)
        return result


class PopulationMLP(nn.Module):
    """Baseline sem atenção que processa os dois tokens do mesmo grupo."""
    def __init__(self, feature_dim: int, latent_dim: int, groups: int, hidden_dim: int = 128):
        super().__init__()
        self.groups = groups
        self.network = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.proportion_head = nn.Linear(hidden_dim, 1)
        self.center_head = nn.Linear(hidden_dim, latent_dim)
        self.dispersion_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor, group_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        del group_ids
        paired = torch.cat([features[:, :self.groups], features[:, self.groups:]], dim=-1)
        hidden = self.network(paired)
        return {"proportion": torch.softmax(self.proportion_head(hidden).squeeze(-1), dim=-1),
                "latent_mean": self.center_head(hidden),
                "latent_dispersion": torch.nn.functional.softplus(self.dispersion_head(hidden))}
