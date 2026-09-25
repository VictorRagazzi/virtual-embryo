"""Build artifact-backed E020 tasks without cell-to-cell pairing."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from src.approaches.population_transformer.data_v2 import official_split
from src.approaches.population_transformer.objectives import hybrid_gene_selection
from src.approaches.population_transformer.pipeline_v2 import fixed_population_draw, fixed_residual_draw
from src.approaches.population_transformer.temporal import state_for_source, token_features
from src.scripts.run_e018_population_transformer import normalized_target, scales


def read_expression(path: Path, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r"); unique, inverse = np.unique(rows, return_inverse=True)
    matrix = data.X[unique][:, columns]
    value = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    data.file.close(); return value[inverse].astype(np.float32)


def source_time(name: str) -> float:
    digits = Path(name).stem.removesuffix("_ex").removeprefix("E")
    return float(f"{digits[0]}.{digits[1:]}") if len(digits) > 1 else float(digits)


def source_rows(path: Path) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r"); rows = np.arange(data.n_obs); data.file.close(); return rows


def source_state(cache, tokens, source: str, local_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    mask = np.flatnonzero(cache["sources"] == source)
    if local_indices is None: return state_for_source(cache, tokens, source)
    indices = mask[local_indices]; latent, labels = cache["latent"][indices], tokens["labels"][indices]
    k = tokens["centers"].shape[0]; counts = np.bincount(labels, minlength=k)
    means = np.zeros((k, latent.shape[1]), np.float32); dispersion = np.zeros_like(means)
    for group in np.flatnonzero(counts):
        values = latent[labels == group]; means[group], dispersion[group] = values.mean(0), values.std(0)
    return {"proportion": (counts / counts.sum()).astype(np.float32), "latent_mean": means, "latent_dispersion": dispersion}


def select_training_genes(paths: list[Path], train_rows: dict[str, np.ndarray], count: int, seed: int,
                          scan_cells: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute hybrid statistics from train rows only; holdouts never enter."""
    rng = np.random.default_rng(seed); sums = squares = first = last = None; total = 0
    means = []
    for path in paths:
        data = ad.read_h5ad(path, backed="r"); allowed = train_rows.get(path.name, np.arange(data.n_obs))
        rows = np.sort(rng.choice(allowed, min(scan_cells, len(allowed)), replace=False)); matrix = data.X[rows]
        xsum = np.asarray(matrix.sum(0)).ravel(); xsq = np.asarray(matrix.multiply(matrix).sum(0)).ravel() if sp.issparse(matrix) else np.square(matrix).sum(0)
        sums = xsum if sums is None else sums + xsum; squares = xsq if squares is None else squares + xsq; total += len(rows)
        means.append(xsum / len(rows)); data.file.close()
    variance = np.maximum(squares / total - (sums / total) ** 2, 0)
    change = means[-1] - means[-2]
    return hybrid_gene_selection(variance, change, count), variance, change


def build_tasks(*, cache, tokens, decoder_data: dict, source_paths: dict[str, Path], sequence: list[str],
                holdout_source: str, official_sources: tuple[str, str], official_holdout: int,
                ranking_genes: int, loss_cells: int, seed: int, device: torch.device) -> tuple[list[dict], dict, dict]:
    """Build external windows plus the D010,D001(train)->D002(train) task."""
    split_rows = {}
    for offset, source in enumerate(official_sources):
        data = ad.read_h5ad(source_paths[source], backed="r")
        if official_holdout:
            train, held = official_split(data.obs["celltype"].to_numpy(), official_holdout, seed + offset)
        else:
            train, held = np.arange(data.n_obs, dtype=np.int64), np.empty(0, dtype=np.int64)
        data.file.close(); split_rows[source] = train; split_rows[source + "_holdout"] = held
    training_sources = [name for name in sequence if name != holdout_source] + list(official_sources)
    gene_indices, _, _ = select_training_genes([source_paths[name] for name in training_sources], split_rows,
                                                ranking_genes, seed)
    state_map = {name: source_state(cache, tokens, name, split_rows.get(name)) for name in sequence + list(official_sources)}
    normalization_states = [state_map[name] for name in sequence if name != holdout_source] + [state_map[official_sources[0]], state_map[official_sources[1]]]
    center_mean, center_scale, dispersion_scale = scales(normalization_states)
    state_dict = decoder_data["state_dict"]
    if "linear.weight" not in state_dict: raise ValueError("E020 currently requires the audited linear E011 decoder.")
    weight = state_dict["linear.weight"][gene_indices].float(); bias = state_dict["linear.bias"][gene_indices].float()
    examples = []
    windows = [(sequence[i-2], sequence[i-1], sequence[i]) for i in range(2, len(sequence)-1)]
    windows.append((sequence[-2], official_sources[0], official_sources[1]))
    rng = np.random.default_rng(seed)
    for number, (first, previous, target_name) in enumerate(windows):
        target_state, previous_state = state_map[target_name], state_map[previous]
        features, group_ids = token_features([state_map[first], previous_state], [source_time(first), source_time(previous)],
                                              source_time(target_name), center_mean, center_scale, dispersion_scale)
        groups = fixed_population_draw(target_state["proportion"], loss_cells, seed + number)
        source_global = np.flatnonzero(cache["sources"] == previous); allowed = split_rows.get(previous)
        if allowed is not None: source_global = source_global[allowed]
        residual, _ = fixed_residual_draw(cache["latent"][source_global], tokens["labels"][source_global], groups,
                                           previous_state, tokens["centers"], seed + number)
        target_path, previous_path = source_paths[target_name], source_paths[previous]
        target_allowed = split_rows.get(target_name, source_rows(target_path))
        previous_allowed = split_rows.get(previous, source_rows(previous_path))
        target_rows = np.sort(rng.choice(target_allowed, loss_cells, replace=len(target_allowed) < loss_cells))
        previous_rows = np.sort(rng.choice(previous_allowed, loss_cells, replace=len(previous_allowed) < loss_cells))
        examples.append({"features": features, "groups": group_ids, "target": target_state,
            "normalized_target": normalized_target(target_state, center_mean, center_scale, dispersion_scale, device),
            "draw_groups": torch.tensor(groups), "residuals": torch.tensor(residual), "decoder_weight": weight,
            "decoder_bias": bias, "target_expression": torch.tensor(read_expression(target_path, target_rows, gene_indices)),
            "reference_expression": torch.tensor(read_expression(previous_path, previous_rows, gene_indices)),
            "normalization": (center_mean, center_scale, dispersion_scale), "sources": [first, previous, target_name]})
    first, previous, target_name = sequence[-3], sequence[-2], holdout_source
    target_state, previous_state = state_map[target_name], state_map[previous]
    features, group_ids = token_features([state_map[first], previous_state], [source_time(first), source_time(previous)],
                                          source_time(target_name), center_mean, center_scale, dispersion_scale)
    groups = fixed_population_draw(previous_state["proportion"], loss_cells, seed + 10_000)
    source_global = np.flatnonzero(cache["sources"] == previous)
    residual, _ = fixed_residual_draw(cache["latent"][source_global], tokens["labels"][source_global], groups,
                                       previous_state, tokens["centers"], seed + 10_000)
    target_rows = np.sort(rng.choice(source_rows(source_paths[target_name]), loss_cells, replace=False))
    previous_rows = np.sort(rng.choice(source_rows(source_paths[previous]), loss_cells, replace=False))
    validation = {"features": features, "groups": group_ids, "target": target_state, "previous_state": previous_state,
        "draw_groups": torch.tensor(groups), "residuals": torch.tensor(residual), "decoder_weight": weight,
        "decoder_bias": bias, "target_expression": torch.tensor(read_expression(source_paths[target_name], target_rows, gene_indices)),
        "reference_expression": torch.tensor(read_expression(source_paths[previous], previous_rows, gene_indices)),
        "normalization": (center_mean, center_scale, dispersion_scale), "sources": [first, previous, target_name]}
    metadata = {"gene_indices": gene_indices, "splits": split_rows, "normalization": (center_mean, center_scale, dispersion_scale)}
    return examples, validation, metadata
