"""Cria subsets seguros e compatíveis para score local do veckit.

Gera os três arquivos com a mesma ordem de células/genes reduzidos:

    uv run python src/scripts/subset_genes.py \
      --pred data/t2_smoke/prediction.h5ad \
      --target data/E95_sample.h5ad \
      --reference data/E85_sample.h5ad \
      --max-cells 500 \
      --max-genes 2000 \
      --seed 42 \
      --out-dir data/t2_smoke/score_subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp


def load_kept_genes(removed_csv: Path, reference_var_names: pd.Index) -> list[str]:
    removed_df = pd.read_csv(removed_csv)
    removed_set = set(removed_df["gene_name"].astype(str))
    return [g for g in reference_var_names.astype(str) if g not in removed_set]


def choose_cells(n_obs: int, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or max_cells >= n_obs:
        return np.arange(n_obs, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=max_cells, replace=False).astype(np.int64))


def choose_genes(
    *,
    target: ad.AnnData,
    reference: ad.AnnData,
    candidate_genes: list[str],
    max_genes: int,
    seed: int,
) -> list[str]:
    if max_genes <= 0 or max_genes >= len(candidate_genes):
        return candidate_genes

    n_var = max_genes // 3
    n_de = max_genes // 3
    n_random = max_genes - n_var - n_de

    target_idx = pd.Index(target.var_names.astype(str)).get_indexer(candidate_genes)
    ref_idx = pd.Index(reference.var_names.astype(str)).get_indexer(candidate_genes)
    if (target_idx < 0).any() or (ref_idx < 0).any():
        raise ValueError("candidate_genes precisam existir em target e reference.")

    target_x = _as_dense(target.X[:, target_idx])
    ref_x = _as_dense(reference.X[:, ref_idx])
    variance = np.var(np.vstack([target_x, ref_x]), axis=0)
    de = np.abs(target_x.mean(axis=0) - ref_x.mean(axis=0))

    selected: list[str] = []
    selected.extend(_top_not_selected(candidate_genes, variance, n_var, selected))
    selected.extend(_top_not_selected(candidate_genes, de, n_de, selected))

    remaining = [g for g in candidate_genes if g not in set(selected)]
    rng = np.random.default_rng(seed)
    if remaining and n_random > 0:
        take = min(n_random, len(remaining))
        selected.extend(sorted(rng.choice(remaining, size=take, replace=False).astype(str)))

    if len(selected) < max_genes:
        selected.extend([g for g in candidate_genes if g not in set(selected)][: max_genes - len(selected)])

    original_order = {g: i for i, g in enumerate(candidate_genes)}
    return sorted(selected[:max_genes], key=original_order.__getitem__)


def subset_and_save(path: Path, genes: list[str], cells: np.ndarray, out_path: Path) -> None:
    a = ad.read_h5ad(path)
    missing = [g for g in genes if g not in set(a.var_names.astype(str))]
    if missing:
        raise ValueError(f"{path}: faltam {len(missing)} genes, primeiros: {missing[:5]}")
    a_subset = a[cells, genes].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_subset.write_h5ad(out_path)
    print(
        f"  {path.name}: {a.n_obs:,}x{a.n_vars:,} -> "
        f"{a_subset.n_obs:,}x{a_subset.n_vars:,} em {out_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--removed-genes", type=Path)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--max-cells", type=int, default=500)
    p.add_argument("--max-genes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data/t2_smoke/score_subset"))
    args = p.parse_args()

    pred = ad.read_h5ad(args.pred)
    target = ad.read_h5ad(args.target)
    reference = ad.read_h5ad(args.reference)

    shared = [
        g
        for g in reference.var_names.astype(str)
        if g in set(target.var_names.astype(str)) and g in set(pred.var_names.astype(str))
    ]
    if args.removed_genes:
        kept = set(load_kept_genes(args.removed_genes, reference.var_names))
        shared = [g for g in shared if g in kept]
    if not shared:
        raise ValueError("pred/target/reference não compartilham genes suficientes.")

    genes = choose_genes(
        target=target,
        reference=reference,
        candidate_genes=shared,
        max_genes=args.max_genes,
        seed=args.seed,
    )
    pred_cells = choose_cells(pred.n_obs, args.max_cells, args.seed)
    target_cells = choose_cells(target.n_obs, args.max_cells, args.seed + 1)
    reference_cells = choose_cells(reference.n_obs, args.max_cells, args.seed + 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_and_save(args.pred, genes, pred_cells, args.out_dir / "pred_subset.h5ad")
    subset_and_save(args.target, genes, target_cells, args.out_dir / "target_subset.h5ad")
    subset_and_save(args.reference, genes, reference_cells, args.out_dir / "reference_subset.h5ad")

    manifest = {
        "format_version": 1,
        "seed": int(args.seed),
        "max_cells": int(args.max_cells),
        "max_genes": int(args.max_genes),
        "n_genes": len(genes),
        "genes": genes,
        "inputs": {
            "pred": str(args.pred),
            "target": str(args.target),
            "reference": str(args.reference),
        },
        "outputs": {
            "pred": str(args.out_dir / "pred_subset.h5ad"),
            "target": str(args.out_dir / "target_subset.h5ad"),
            "reference": str(args.out_dir / "reference_subset.h5ad"),
        },
    }
    (args.out_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nPronto. Arquivos em: {args.out_dir.resolve()}")


def _as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _top_not_selected(
    genes: list[str],
    scores: np.ndarray,
    n: int,
    selected: list[str],
) -> list[str]:
    if n <= 0:
        return []
    used = set(selected)
    out = []
    for i in np.argsort(-scores, kind="stable"):
        gene = genes[int(i)]
        if gene in used:
            continue
        out.append(gene)
        used.add(gene)
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    main()
