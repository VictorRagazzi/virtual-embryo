"""
Aplica a MESMA lista de genes (saída do filtro de baixa variância em
pre_processement.py, calculado sobre E8.5+E9.5 juntos) em três arquivos
.h5ad -- predição, target e reference -- para que o score_h5ad.py do
veckit não rejeite por mismatch de var_names, e para reduzir o custo de
memória de qualquer métrica quadrática em número de genes.

Uso:
    uv run python src/scripts/subset_genes.py \\
        --removed-genes data/preprocessing/joint_removed_genes.csv \\
        --pred data/prediction_e9_5_decoder_val.h5ad \\
        --target data/E95.h5ad \\
        --reference data/E85.h5ad \\
        --out-dir data/scored_subset

Gera:
    data/scored_subset/pred_subset.h5ad
    data/scored_subset/target_subset.h5ad
    data/scored_subset/reference_subset.h5ad

Depois rode o score normalmente apontando pra esses 3 arquivos:
    uv run python -m src.scgpt_decoder.main score \\
        --score-input data/scored_subset/pred_subset.h5ad \\
        --e85 data/scored_subset/reference_subset.h5ad \\
        --e95 data/scored_subset/target_subset.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def load_kept_genes(removed_csv: Path, reference_var_names: pd.Index) -> list[str]:
    """A lista de mantidos = todos os genes do dataset original MENOS os que
    aparecem no CSV de removidos (que lista só os removidos, não os mantidos)."""
    removed_df = pd.read_csv(removed_csv)
    removed_set = set(removed_df["gene_name"].astype(str))
    kept = [g for g in reference_var_names.astype(str) if g not in removed_set]
    return kept


def subset_and_save(path: Path, kept_genes: list[str], out_path: Path) -> None:
    a = ad.read_h5ad(path)
    var_names = a.var_names.astype(str)

    missing = [g for g in kept_genes if g not in set(var_names)]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} genes da lista de mantidos não existem "
            f"neste arquivo (ex: {missing[:5]}). Os arquivos pred/target/reference "
            f"precisam compartilhar o mesmo painel completo de genes original."
        )

    a_subset = a[:, kept_genes].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_subset.write_h5ad(out_path)
    print(f"  {path.name}: {a.n_vars:,} -> {a_subset.n_vars:,} genes, "
          f"{a_subset.n_obs:,} células -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--removed-genes", type=Path, required=True,
                   help="CSV gerado por filter_low_variance (coluna 'gene_name')")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/scored_subset"))
    args = p.parse_args()

    # usa o --reference (E8.5 completo, 32285 genes) como universo de genes
    # original para decidir a lista de "mantidos" a partir do CSV de removidos
    ref_full = ad.read_h5ad(args.reference)
    kept_genes = load_kept_genes(args.removed_genes, ref_full.var_names)
    print(f"Genes mantidos: {len(kept_genes):,} de {ref_full.n_vars:,} originais")
    del ref_full

    print("\nAplicando subset consistente nos 3 arquivos...")
    subset_and_save(args.pred, kept_genes, args.out_dir / "pred_subset.h5ad")
    subset_and_save(args.target, kept_genes, args.out_dir / "target_subset.h5ad")
    subset_and_save(args.reference, kept_genes, args.out_dir / "reference_subset.h5ad")

    print(f"\nPronto. Arquivos em: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()