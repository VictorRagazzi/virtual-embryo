"""
Compara dois AnnData (.h5ad) e reporta estatísticas gerais + por gene/por
célula sobre as diferenças entre eles.

Pensado principalmente para o caso deste projeto:

    uv run python -m src.scripts.compare_predictions \
        --a data/E95.h5ad \
        --b outputs/t1_temporal/e10_5_predicted.h5ad \
        --a-label "E9.5 (real)" \
        --b-label "E10.5 (predito)"

Mas funciona para qualquer par de .h5ad com genes em comum (não precisa ser
exatamente o mesmo conjunto de genes nem de células -- o script alinha pela
interseção de var_names e assume que a ordem das células é comparável quando
os dois arquivos têm o mesmo n_obs; ver `--match-cells-by`).

O que é reportado:
    1. Estatísticas globais (min/max/mean/std/%zeros/%negativos) de cada
       dataset e do delta (B - A).
    2. Se existir `var['t1_gene_source']` em B (saída do predict.py com
       --full-genes), separa a análise em genes 'predicted' vs
       'passthrough_e95'.
    3. Top genes com maior variação média (|delta| médio entre células) --
       útil pra ver quais genes o modelo mais "mexeu".
    4. Se `--celltype-col` for passado e existir em ambos, quebra a análise
       por tipo celular.
    5. Amostra dos casos mais extremos (maior |delta| célula x gene
       individual), útil para investigar valores suspeitos (ex.: os
       negativos que geraram erro na submissão).

Não depende de nada de plotting -- é só texto no console, para rodar rápido
em qualquer ambiente (inclusive sem GPU/sem display).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _dense(X) -> np.ndarray:
    return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)


def _print_stats(label: str, X: np.ndarray) -> None:
    logger.info(
        "%-28s min=%9.4f  max=%9.4f  mean=%8.4f  std=%8.4f  %%zeros=%6.2f%%  %%neg=%6.2f%%",
        label, X.min(), X.max(), X.mean(), X.std(),
        100 * (X == 0).mean(), 100 * (X < 0).mean(),
    )


def align_common_genes(a: ad.AnnData, b: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    common = [g for g in a.var_names if g in set(b.var_names)]
    if len(common) == 0:
        raise ValueError("Nenhum gene em comum entre os dois arquivos.")
    if len(common) < min(a.n_vars, b.n_vars):
        logger.warning(
            "Apenas %d genes em comum (A tem %d, B tem %d) -- comparando só a interseção.",
            len(common), a.n_vars, b.n_vars,
        )
    return a[:, common], b[:, common], common


def check_cell_alignment(a: ad.AnnData, b: ad.AnnData, match_cells_by: str | None) -> None:
    if a.n_obs != b.n_obs:
        raise ValueError(
            f"Datasets têm números de células diferentes (A={a.n_obs}, B={b.n_obs}). "
            "Célula i de A é comparada com célula i de B por posição -- eles precisam "
            "estar alinhados (mesmo subsample/ordem) para essa comparação fazer sentido."
        )
    if match_cells_by is not None:
        if match_cells_by not in a.obs.columns or match_cells_by not in b.obs.columns:
            logger.warning(
                "Coluna '%s' não encontrada em um dos dois .obs -- pulando checagem de alinhamento.",
                match_cells_by,
            )
            return
        ids_a = a.obs[match_cells_by].to_numpy()
        ids_b = b.obs[match_cells_by].to_numpy()
        if not np.array_equal(ids_a, ids_b):
            n_mismatch = int((ids_a != ids_b).sum())
            logger.warning(
                "%d/%d células não batem em '%s' entre A e B na mesma posição -- "
                "a comparação célula-a-célula pode não ser válida.",
                n_mismatch, len(ids_a), match_cells_by,
            )
        else:
            logger.info("Alinhamento de células verificado via '%s': OK.", match_cells_by)


def top_genes_by_delta(gene_names: list[str], delta: np.ndarray, n: int = 20) -> pd.DataFrame:
    mean_abs_delta = np.abs(delta).mean(axis=0)
    mean_delta = delta.mean(axis=0)
    order = np.argsort(-mean_abs_delta)[:n]
    return pd.DataFrame({
        "gene": [gene_names[i] for i in order],
        "mean_abs_delta": mean_abs_delta[order],
        "mean_delta": mean_delta[order],
    })


def most_extreme_cases(
    gene_names: list[str],
    Xa: np.ndarray,
    Xb: np.ndarray,
    n: int = 15,
    only_negative_b: bool = False,
) -> pd.DataFrame:
    delta = Xb - Xa
    mask = (Xb < 0) if only_negative_b else np.ones_like(delta, dtype=bool)
    if not mask.any():
        return pd.DataFrame(columns=["cell_idx", "gene", "A", "B", "delta"])

    flat_delta = np.where(mask, np.abs(delta), -np.inf)
    flat_idx = np.argsort(flat_delta.ravel())[::-1][:n]
    cell_idx, gene_idx = np.unravel_index(flat_idx, delta.shape)
    return pd.DataFrame({
        "cell_idx": cell_idx,
        "gene": [gene_names[g] for g in gene_idx],
        "A": Xa[cell_idx, gene_idx],
        "B": Xb[cell_idx, gene_idx],
        "delta": delta[cell_idx, gene_idx],
    })


def compare_by_celltype(
    a: ad.AnnData, b: ad.AnnData, celltype_col: str, Xa: np.ndarray, Xb: np.ndarray
) -> None:
    if celltype_col not in a.obs.columns:
        logger.warning("Coluna '%s' não encontrada em A -- pulando quebra por tipo celular.", celltype_col)
        return
    labels = a.obs[celltype_col].to_numpy()
    logger.info("-" * 70)
    logger.info("Quebra por '%s'", celltype_col)
    logger.info("-" * 70)
    for label in pd.unique(labels):
        mask = labels == label
        delta = Xb[mask] - Xa[mask]
        logger.info(
            "  %-25s n=%6d  mean_A=%7.4f  mean_B=%7.4f  mean_delta=%7.4f  %%neg_B=%6.2f%%",
            str(label), mask.sum(), Xa[mask].mean(), Xb[mask].mean(), delta.mean(),
            100 * (Xb[mask] < 0).mean(),
        )


def run_compare(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    celltype_col: str | None,
    match_cells_by: str | None,
    top_n_genes: int,
    top_n_extreme: int,
) -> None:
    logger.info("Lendo A: %s", path_a)
    a = ad.read_h5ad(path_a)
    logger.info("Lendo B: %s", path_b)
    b = ad.read_h5ad(path_b)

    check_cell_alignment(a, b, match_cells_by)
    a, b, gene_names = align_common_genes(a, b)

    Xa = _dense(a.X).astype(np.float32)
    Xb = _dense(b.X).astype(np.float32)
    delta = Xb - Xa

    logger.info("=" * 70)
    logger.info("COMPARAÇÃO GERAL: %s (A) vs %s (B)", label_a, label_b)
    logger.info("=" * 70)
    logger.info("Genes em comum: %d | Células: %d", len(gene_names), a.n_obs)
    _print_stats(f"{label_a} (A)", Xa)
    _print_stats(f"{label_b} (B)", Xb)
    _print_stats("Delta (B - A)", delta)

    # se B tiver a marcação de origem por gene (saída do predict.py com --full-genes)
    if "t1_gene_source" in b.var.columns:
        logger.info("-" * 70)
        logger.info("Quebra por origem do gene em B (var['t1_gene_source'])")
        logger.info("-" * 70)
        source = b.var["t1_gene_source"].to_numpy()
        for lab in np.unique(source):
            cols = source == lab
            logger.info("  Grupo '%s' (%d genes):", lab, cols.sum())
            _print_stats(f"    {label_a}", Xa[:, cols])
            _print_stats(f"    {label_b}", Xb[:, cols])
            _print_stats("    Delta", delta[:, cols])

    logger.info("-" * 70)
    logger.info("Top %d genes com maior |delta| médio entre células", top_n_genes)
    logger.info("-" * 70)
    top_df = top_genes_by_delta(gene_names, delta, top_n_genes)
    for _, row in top_df.iterrows():
        logger.info("  %-20s mean_abs_delta=%8.4f  mean_delta=%8.4f", row["gene"], row["mean_abs_delta"], row["mean_delta"])

    n_neg_b = int((Xb < 0).sum())
    if n_neg_b > 0:
        logger.info("-" * 70)
        logger.info(
            "%d valores negativos em B (%.4f%% do total) -- top %d casos mais extremos:",
            n_neg_b, 100 * n_neg_b / Xb.size, top_n_extreme,
        )
        logger.info("-" * 70)
        extreme_df = most_extreme_cases(gene_names, Xa, Xb, top_n_extreme, only_negative_b=True)
        for _, row in extreme_df.iterrows():
            logger.info(
                "  gene=%-20s cell=%-6d  A=%8.4f  B=%8.4f  delta=%8.4f",
                row["gene"], row["cell_idx"], row["A"], row["B"], row["delta"],
            )
    else:
        logger.info("Nenhum valor negativo em B.")

    if celltype_col is not None:
        compare_by_celltype(a, b, celltype_col, Xa, Xb)

    logger.info("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="Ex.: data/E95.h5ad")
    p.add_argument("--b", type=Path, required=True, help="Ex.: outputs/t1_temporal/e10_5_predicted.h5ad")
    p.add_argument("--a-label", type=str, default="A")
    p.add_argument("--b-label", type=str, default="B")
    p.add_argument("--celltype-col", type=str, default=None, help="ex.: celltype -- quebra a análise por tipo celular")
    p.add_argument(
        "--match-cells-by", type=str, default=None,
        help="coluna de obs usada só para VERIFICAR que a célula i de A é a mesma célula i de B "
             "(ex.: um id de célula/barcode). Opcional -- se não passar, assume-se alinhamento por posição.",
    )
    p.add_argument("--top-n-genes", type=int, default=20)
    p.add_argument("--top-n-extreme", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_compare(
        args.a, args.b, args.a_label, args.b_label,
        args.celltype_col, args.match_cells_by,
        args.top_n_genes, args.top_n_extreme,
    )