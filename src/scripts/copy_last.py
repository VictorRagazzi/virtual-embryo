"""Gera o baseline ``copy_last``: uma cópia sem alteração da expressão E9.5."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.m0_contract import validate_task1_output


def build_copy_last(source: ad.AnnData, seed: int, target_cells: int | None) -> ad.AnnData:
    """Copia ``.X`` de E9.5, opcionalmente com subamostragem sem reposição."""
    if target_cells is not None:
        if target_cells <= 0 or target_cells > source.n_obs:
            raise ValueError("--target-cells deve estar entre 1 e o número de células de E9.5.")
        indices = np.random.default_rng(seed).choice(source.n_obs, target_cells, replace=False)
        matrix = source[indices].X.copy()
    else:
        matrix = source.X.copy()
    obs = pd.DataFrame(index=[f"copy_last_{index:06d}" for index in range(matrix.shape[0])])
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=source.var_names.copy()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/E95.h5ad"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ad.read_h5ad(args.input)
    output = build_copy_last(source, args.seed, args.target_cells)
    metrics = validate_task1_output(output, source.var_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output)
    print({"baseline": "copy_last", "seed": args.seed, "output": str(args.output), **metrics})


if __name__ == "__main__":
    main()
