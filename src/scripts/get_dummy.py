"""Gera uma submissão dummy estruturalmente válida, com expressão toda zero."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.m0_contract import validate_task1_output


def build_dummy(reference: ad.AnnData, n_cells: int) -> ad.AnnData:
    if n_cells <= 0:
        raise ValueError("--n-cells deve ser positivo.")
    matrix = sparse.csr_matrix((n_cells, reference.n_vars), dtype=np.float32)
    obs = pd.DataFrame(index=[f"dummy_{index:06d}" for index in range(n_cells)])
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=reference.var_names.copy()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/E95.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("submission_dummy.h5ad"))
    parser.add_argument("--n-cells", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42, help="Registrada; o dummy é determinístico.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = ad.read_h5ad(args.reference, backed="r")
    try:
        output = build_dummy(reference, args.n_cells)
        metrics = validate_task1_output(output, reference.var_names)
    finally:
        reference.file.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output)
    print({"baseline": "dummy_zero", "seed": args.seed, "output": str(args.output), **metrics})


if __name__ == "__main__":
    main()
