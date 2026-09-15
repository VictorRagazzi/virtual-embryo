"""
Subsample estratificado por tipo celular (preserva as proporções de
obs[--celltype-col]).

Útil para reduzir o `.h5ad` de predição (ex.: 17k células) para um tamanho
manejável mantendo a composição de tipos celulares aproximadamente igual à
original -- em vez de um sample aleatório simples, que pode sub-representar
tipos raros.

Uso:

uv run python -m src.scripts.subsample --input-h5ad data/E95.h5ad --output-h5ad data/E95_sub2000.h5ad --n-cells 2000 --celltype-col celltype

    uv run python -m src.scripts.subsample \
        --input-h5ad outputs/t1_temporal/e10_5_predicted.h5ad \
        --output-h5ad outputs/t1_temporal/e10_5_predicted_sub2000.h5ad \
        --n-cells 2000 \
        --celltype-col celltype

Também aceita --frac 0.1 no lugar de --n-cells.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def allocate_counts_by_proportion(group_sizes: pd.Series, n_target: int) -> pd.Series:
    """
    Distribui n_target células entre os grupos proporcionalmente ao tamanho
    original de cada grupo, usando o método do "maior resto" para garantir que
    a soma final seja exatamente n_target (e nunca aloque mais que o grupo tem).
    """
    total = group_sizes.sum()
    if n_target >= total:
        return group_sizes.copy()

    raw = group_sizes / total * n_target
    base = np.floor(raw).astype(int)
    base = np.minimum(base, group_sizes)  # nunca pedir mais do que existe

    remainder = n_target - base.sum()
    # distribui o restante para os grupos com maior parte fracionária
    # (respeitando o teto = tamanho do grupo)
    frac = (raw - base).sort_values(ascending=False)
    for group in frac.index:
        if remainder <= 0:
            break
        if base[group] < group_sizes[group]:
            base[group] += 1
            remainder -= 1

    # se ainda sobrar (todos os grupos saturados), não há o que fazer -- já
    # está no máximo possível (n_target > total, tratado acima, então isso
    # só ocorreria em casos degenerados)
    return base


def stratified_subsample(
    adata: ad.AnnData,
    celltype_col: str,
    n_cells: int | None,
    frac: float | None,
    seed: int,
    min_per_group: int = 0,
) -> ad.AnnData:
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Coluna '{celltype_col}' não encontrada em obs. "
            f"Colunas disponíveis: {list(adata.obs.columns)}"
        )

    group_sizes = adata.obs[celltype_col].value_counts()
    total = int(group_sizes.sum())

    if n_cells is None and frac is None:
        raise ValueError("Forneça --n-cells ou --frac.")
    if n_cells is None:
        n_cells = int(round(total * frac))
    n_cells = min(n_cells, total)

    allocation = allocate_counts_by_proportion(group_sizes, n_cells)

    if min_per_group > 0:
        # garante um mínimo por grupo presente (útil pra tipos raros não
        # sumirem completamente), sem exceder o tamanho do grupo nem o total
        deficit_groups = allocation[(allocation < min_per_group) & (group_sizes >= min_per_group)]
        for group in deficit_groups.index:
            allocation[group] = min_per_group
        # reduz proporcionalmente os grupos maiores para compensar o excesso,
        # se necessário, para manter a soma perto de n_cells
        excess = allocation.sum() - n_cells
        if excess > 0:
            reducible = allocation[~allocation.index.isin(deficit_groups.index)].sort_values(ascending=False)
            for group in reducible.index:
                if excess <= 0:
                    break
                cut = min(excess, allocation[group])
                allocation[group] -= cut
                excess -= cut

    rng = np.random.default_rng(seed)
    chosen_idx = []
    for group, n in allocation.items():
        if n <= 0:
            continue
        group_idx = np.flatnonzero(adata.obs[celltype_col].to_numpy() == group)
        pick = rng.choice(group_idx, size=int(n), replace=False)
        chosen_idx.append(pick)
    chosen_idx = np.sort(np.concatenate(chosen_idx))

    logger.info("Total original: %d células. Subsample: %d células (%d grupos).",
                total, len(chosen_idx), len(allocation[allocation > 0]))
    logger.info("\nProporções originais (%%):\n%s",
                (group_sizes / total * 100).round(2).to_string())
    sub = adata[chosen_idx].copy()
    logger.info("\nProporções no subsample (%%):\n%s",
                (sub.obs[celltype_col].value_counts() / len(sub) * 100).round(2).to_string())

    return sub


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-h5ad", type=Path, required=True)
    p.add_argument("--output-h5ad", type=Path, required=True)
    p.add_argument("--celltype-col", type=str, default="celltype")
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--frac", type=float, default=None)
    p.add_argument("--min-per-group", type=int, default=0,
                    help="mínimo de células por tipo celular presente, quando o grupo tiver pelo menos esse tamanho.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    sub = stratified_subsample(
        adata, args.celltype_col, args.n_cells, args.frac, args.seed, args.min_per_group
    )
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(args.output_h5ad)
    logger.info("Salvo em %s", args.output_h5ad)