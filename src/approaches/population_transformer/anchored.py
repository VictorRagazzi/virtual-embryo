"""Expression-anchored population generation for E021.

The temporal model predicts population changes, while real cells from the last
observed stage supply single-cell covariance and sparsity. No cell-to-cell
correspondence with the target stage is assumed.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


def balanced_source_indices(
    target_groups: np.ndarray,
    source_groups: np.ndarray,
    source_centers: np.ndarray,
    target_centers: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Draw group-wise, exhausting every source pool before repetition."""
    target_groups = np.asarray(target_groups, dtype=np.int64)
    source_groups = np.asarray(source_groups, dtype=np.int64)
    if source_centers.ndim != 2 or target_centers.shape != source_centers.shape:
        raise ValueError("source_centers and target_centers must have the same 2D shape")
    if len(source_groups) == 0:
        raise ValueError("source population is empty")
    rng = np.random.default_rng(seed)
    pools = {int(group): np.flatnonzero(source_groups == group) for group in np.unique(source_groups)}
    available = np.asarray(sorted(pools), dtype=np.int64)
    chosen = np.empty(len(target_groups), dtype=np.int64)
    fallback = 0
    for group in np.unique(target_groups):
        output = np.flatnonzero(target_groups == group)
        source_group = int(group)
        if source_group not in pools:
            distance = np.square(source_centers[available] - target_centers[source_group]).sum(axis=1)
            source_group = int(available[np.argmin(distance)])
            fallback += len(output)
        pool = pools[source_group]
        cycles = int(np.ceil(len(output) / len(pool)))
        draws = np.concatenate([rng.permutation(pool) for _ in range(cycles)])
        chosen[output] = draws[: len(output)]
    return chosen, fallback


def weighted_population_without_replacement(
    proportion: np.ndarray, source_groups: np.ndarray, n_cells: int, seed: int
) -> np.ndarray:
    """Select unique anchors while approximating requested group proportions."""
    source_groups = np.asarray(source_groups, dtype=np.int64)
    proportion = np.asarray(proportion, dtype=np.float64)
    if not 0 < n_cells <= len(source_groups):
        raise ValueError("n_cells must be positive and no larger than the source population")
    counts = np.bincount(source_groups, minlength=len(proportion))
    weights = np.zeros(len(source_groups), dtype=np.float64)
    observed = counts[source_groups] > 0
    weights[observed] = proportion[source_groups[observed]] / counts[source_groups[observed]]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("requested proportions have no support in the source population")
    weights /= weights.sum()
    return np.sort(np.random.default_rng(seed).choice(len(source_groups), n_cells, replace=False, p=weights))


def fit_gene_programs(change_matrix: np.ndarray, n_programs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit low-rank signed gene programs to known group/time changes."""
    changes = np.asarray(change_matrix, dtype=np.float32)
    if changes.ndim != 2 or not np.isfinite(changes).all():
        raise ValueError("change_matrix must be a finite 2D matrix")
    maximum = min(changes.shape)
    if not 0 < n_programs <= maximum:
        raise ValueError(f"n_programs must be in [1, {maximum}]")
    svd = TruncatedSVD(n_components=n_programs, random_state=seed)
    coefficients = svd.fit_transform(changes).astype(np.float32)
    return coefficients, svd.components_.astype(np.float32)


def hybrid_gene_programs(changes, pseudobulk_changes, n_general, n_de, seed):
    """Append training-only, sign-enriched residual directions to a fixed SVD."""
    _, general = fit_gene_programs(changes, n_general, seed)
    if not n_de:
        return general
    pb = np.asarray(pseudobulk_changes, dtype=np.float32)
    stability = np.abs(np.sign(pb).mean(axis=0))
    enriched = np.concatenate([
        changes, pb,
        np.maximum(changes, 0) * stability,
        np.minimum(changes, 0) * stability,
    ])
    residual = enriched - (enriched @ general.T) @ general
    _, extra = fit_gene_programs(residual, n_de, seed)
    # Reorthogonalize in double precision without rotating the original basis.
    rows = [row.astype(np.float64) for row in general]
    for candidate in extra:
        row = candidate.astype(np.float64)
        for _ in range(2):
            for previous in rows:
                row -= np.dot(row, previous) * previous
        norm = np.linalg.norm(row)
        if norm < 1e-6:
            raise ValueError("Insufficient residual rank for requested DE programs")
        rows.append(row / norm)
    return np.asarray(rows, dtype=np.float32)


def anchored_expression(
    base_expression: np.ndarray,
    groups: np.ndarray,
    program_coefficients: np.ndarray,
    programs: np.ndarray,
    *,
    scale: float = 1.0,
    activation_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply group program changes while preserving unsupported zeros."""
    base = np.asarray(base_expression, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    coefficients = np.asarray(program_coefficients, dtype=np.float32)
    programs = np.asarray(programs, dtype=np.float32)
    if base.ndim != 2 or len(base) != len(groups):
        raise ValueError("base_expression and groups have incompatible shapes")
    if coefficients.ndim != 2 or programs.ndim != 2 or coefficients.shape[1] != programs.shape[0]:
        raise ValueError("program coefficient/component shapes are incompatible")
    if coefficients.shape[0] <= (int(groups.max()) if len(groups) else -1) or programs.shape[1] != base.shape[1]:
        raise ValueError("group or gene dimension is incompatible")
    delta = (coefficients[groups] @ programs) * np.float32(scale)
    allowed = base > 0
    if activation_mask is not None:
        activation = np.asarray(activation_mask, dtype=bool)
        if activation.shape != (coefficients.shape[0], base.shape[1]):
            raise ValueError("activation_mask must have shape [groups, genes]")
        allowed |= activation[groups] & (delta > 0)
    result = base + np.where(allowed, delta, 0)
    return np.maximum(result, 0).astype(np.float32)
