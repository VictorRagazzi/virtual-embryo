"""Approximate nonlinear PCA and population-level temporal models for E027."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import nn


class ApproximateKernelPCA:
    """RBF kernel PCA approximated by Nyström features, with a linear pre-image."""

    def __init__(self, n_landmarks: int, n_components: int, gamma: float, ridge: float, seed: int):
        self.kernel = Nystroem(kernel="rbf", gamma=gamma, n_components=n_landmarks, random_state=seed)
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.ridge = float(ridge)
        self.decoder_: np.ndarray | None = None

    def fit_embedding(self, expression: np.ndarray) -> "ApproximateKernelPCA":
        features = self.kernel.fit_transform(np.asarray(expression, dtype=np.float32))
        self.pca.fit(features)
        return self

    def transform(self, expression: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.kernel.transform(np.asarray(expression, dtype=np.float32))).astype(np.float32)

    def fit_decoder(self, latent_blocks: list[np.ndarray], expression_blocks: list[np.ndarray]) -> None:
        dim = latent_blocks[0].shape[1]
        gram = np.zeros((dim + 1, dim + 1), dtype=np.float64)
        cross = np.zeros((dim + 1, expression_blocks[0].shape[1]), dtype=np.float64)
        for latent, expression in zip(latent_blocks, expression_blocks, strict=True):
            design = np.column_stack([latent, np.ones(len(latent), dtype=np.float32)]).astype(np.float64)
            gram += design.T @ design
            cross += design.T @ np.asarray(expression, dtype=np.float64)
        gram.flat[:: dim + 2] += self.ridge
        gram[-1, -1] -= self.ridge
        self.decoder_ = np.linalg.solve(gram, cross).astype(np.float32)

    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        if self.decoder_ is None:
            raise RuntimeError("fit_decoder must be called first")
        design = np.column_stack([latent, np.ones(len(latent), dtype=np.float32)])
        return np.maximum(design @ self.decoder_, 0).astype(np.float32)


def decode_variants(model: ApproximateKernelPCA, predicted: np.ndarray, source_latent: np.ndarray,
                    source_expression: np.ndarray) -> dict[str, np.ndarray]:
    """Decode directly, add an empirical PCA residual, or copy the nearest cell."""
    nearest = NearestNeighbors(n_neighbors=1).fit(source_latent).kneighbors(predicted, return_distance=False).ravel()
    raw = model.inverse_transform(predicted)
    source_raw = model.inverse_transform(source_latent[nearest])
    residual = np.maximum(raw + source_expression[nearest] - source_raw, 0).astype(np.float32)
    return {"decoder_raw": raw, "decoder_residual": residual,
            "nearest_neighbor": source_expression[nearest].astype(np.float32, copy=True)}


def decode_zero_weighted_change(model, predicted, source_latent, source_expression,
                                source_groups, group_support, beta):
    """Apply full decoded change on expressed entries and shrink new activations."""
    nearest = NearestNeighbors(n_neighbors=1).fit(source_latent).kneighbors(predicted, return_distance=False).ravel()
    anchor = np.asarray(source_expression[nearest], dtype=np.float32)
    change = model.inverse_transform(predicted) - model.inverse_transform(source_latent[nearest])
    output = np.maximum(anchor + change, 0)
    zero = anchor == 0
    supported = group_support[np.asarray(source_groups, dtype=np.int64)[nearest]]
    activation = np.float32(beta) * np.maximum(change, 0)
    output[zero] = np.where(supported[zero], activation[zero], 0)
    return output.astype(np.float32)


class CellSetTransformer(nn.Module):
    """Predict a future population from individual cells without cell pairing."""

    def __init__(self, latent_dim: int, model_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(latent_dim + 1, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 2, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.delta = nn.Linear(model_dim, latent_dim)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        minus_two = torch.full((*first.shape[:-1], 1), -2.0, device=first.device)
        minus_one = torch.full((*second.shape[:-1], 1), -1.0, device=second.device)
        values = torch.cat([torch.cat([minus_two, first], -1), torch.cat([minus_one, second], -1)], 1)
        hidden = self.encoder(self.projection(values))[:, -second.shape[1]:]
        return second + self.delta(hidden)


class GroupSummaryTransformer(nn.Module):
    """Baseline that consumes proportion, mean and dispersion per K-means group."""

    def __init__(self, latent_dim: int, model_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection = nn.Linear(2 * latent_dim + 2, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 2, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.delta = nn.Linear(model_dim, latent_dim)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        minus_two = torch.full((*first.shape[:-1], 1), -2.0, device=first.device)
        minus_one = torch.full((*second.shape[:-1], 1), -1.0, device=second.device)
        values = torch.cat([torch.cat([minus_two, first], -1), torch.cat([minus_one, second], -1)], 1)
        hidden = self.encoder(self.projection(values))[:, -second.shape[1]:]
        return second[..., 1:1 + self.latent_dim] + self.delta(hidden)


class SingleStageCellTransformer(nn.Module):
    """Transform one unpaired source population into a target population."""

    def __init__(self, latent_dim: int, model_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(latent_dim, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 2, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.delta = nn.Linear(model_dim, latent_dim)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source + self.delta(self.encoder(self.projection(source)))


class GroupTransitionTransformer(nn.Module):
    """Map one shared K-means population state to the next state."""

    def __init__(self, latent_dim: int, groups: int, model_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.groups = groups
        self.projection = nn.Linear(2 * latent_dim + 2, model_dim)
        self.group_embedding = nn.Embedding(groups, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 2, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.proportion = nn.Linear(model_dim, 1)
        self.center_delta = nn.Linear(model_dim, latent_dim)
        self.dispersion = nn.Linear(model_dim, latent_dim)

    def forward(self, state: torch.Tensor, group_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(self.projection(state) + self.group_embedding(group_ids))
        center = state[..., 1:1 + self.center_delta.out_features]
        return {"proportion": torch.softmax(self.proportion(hidden).squeeze(-1), dim=-1),
                "latent_mean": center + self.center_delta(hidden),
                "latent_dispersion": torch.nn.functional.softplus(self.dispersion(hidden))}


def distribution_loss(prediction: torch.Tensor, target: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant mean, dispersion and sliced-Wasserstein loss."""
    moment = (prediction.mean(1) - target.mean(1)).square().mean()
    moment = moment + (prediction.std(1) - target.std(1)).square().mean()
    pred_projection = torch.sort(prediction @ projections.T, dim=1).values
    target_projection = torch.sort(target @ projections.T, dim=1).values
    return moment + (pred_projection - target_projection).square().mean()


def grouped_latent(latent: np.ndarray, clusterer: MiniBatchKMeans) -> tuple[np.ndarray, np.ndarray]:
    labels = clusterer.predict(latent)
    centers = np.zeros_like(clusterer.cluster_centers_, dtype=np.float32)
    for group in range(len(centers)):
        members = latent[labels == group]
        centers[group] = members.mean(0) if len(members) else clusterer.cluster_centers_[group]
    return centers, labels
