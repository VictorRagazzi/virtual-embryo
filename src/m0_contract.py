"""Contrato estrutural mínimo para matrizes de saída da Task 1."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp


def _stored_values(matrix):
    """Retorna valores materializados; zeros implícitos são seguros."""
    if hasattr(matrix, "_data"):
        return matrix._data
    if sp.issparse(matrix):
        values = getattr(matrix, "data", None)
        return values
    return np.asarray(matrix).ravel()


def _value_stats(values) -> tuple[int, bool, bool, float, float]:
    """Calcula estatísticas de ndarray ou ``SparseDataset`` em blocos."""
    stored = int(values.shape[0])
    minimum, maximum, finite, negative = np.inf, -np.inf, True, False
    for start in range(0, stored, 5_000_000):
        block = np.asarray(values[start : start + 5_000_000])
        if block.size:
            finite = finite and bool(np.isfinite(block).all())
            negative = negative or bool((block < 0).any())
            minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
    return stored, finite, negative, (float(minimum) if stored else 0.0), (float(maximum) if stored else 0.0)


def validate_task1_output(
    adata: ad.AnnData,
    expected_genes: Sequence[str],
    *,
    min_cells: int | None = None,
    max_cells: int | None = None,
) -> dict[str, int | float | str]:
    """Valida ordem gênica, tipo e domínio numérico sem densificar esparsas."""
    expected = np.asarray(expected_genes, dtype=str)
    actual = adata.var_names.to_numpy(dtype=str)
    if adata.n_vars != len(expected) or not np.array_equal(actual, expected):
        raise ValueError("A ordem/identidade dos genes não coincide com a referência.")
    if min_cells is not None and adata.n_obs < min_cells:
        raise ValueError(f"A saída tem {adata.n_obs} células; mínimo exigido: {min_cells}.")
    if max_cells is not None and adata.n_obs > max_cells:
        raise ValueError(f"A saída tem {adata.n_obs} células; máximo exigido: {max_cells}.")
    matrix = adata.X
    if matrix.dtype != np.float32:
        raise ValueError(f".X deve ser float32; recebido {matrix.dtype}.")
    stored, finite, negative, minimum, maximum = _value_stats(_stored_values(matrix))
    if not finite:
        raise ValueError(".X contém NaN ou infinito.")
    if negative:
        raise ValueError(".X contém valores negativos.")
    total = adata.n_obs * adata.n_vars
    return {
        "shape": f"{adata.n_obs}x{adata.n_vars}",
        "dtype": str(matrix.dtype),
        "stored_values": stored,
        "stored_density": float(stored / total) if total else 0.0,
        "min_stored": minimum,
        "max_stored": maximum,
    }
