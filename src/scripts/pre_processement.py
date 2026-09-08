"""
pre_processement.py – Pré-processamento de expressão gênica para o Virtual Embryo Challenge.

Funções principais:
  1. filter_low_variance  – Remove genes com variação abaixo de um limiar
                           e salva a tabela de genes removidos.
  2. group_correlated_genes – Agrupa genes altamente correlacionados e
                              retorna um dicionário de grupos com a
                              correlação mínima do grupo.

Uso rápido:
    uv run python src/scripts/pre_processement.py --e85 data/E85.h5ad --e95 data/E95.h5ad
    uv run python src/scripts/pre_processement.py --e85 data/E85.h5ad --e95 data/E95.h5ad \\
        --variance-threshold 0.01 --correlation-threshold 0.95

Argumentos opcionais:
    --variance-threshold   Variância mínima para manter o gene (default: 0.01)
    --correlation-threshold Limiar de correlação de Pearson para agrupar genes (default: 0.95)
    --output-dir           Pasta para salvar tabelas (default: data/preprocessing/)
    --sample-cells         Número de células a amostrar por dataset (default: 5000, 0 = todos)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


# ─── Constantes padrão ────────────────────────────────────────────────────────

DEFAULT_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_OUTPUT_DIR = Path("data/preprocessing")
DEFAULT_SAMPLE_CELLS = 5_000  # 0 → usa todas as células


# ─── Carregamento ─────────────────────────────────────────────────────────────

def load_dataset(path: Path, label: str, sample_cells: int = 0) -> ad.AnnData:
    """
    Carrega um arquivo .h5ad e, opcionalmente, amostra um subconjunto de células.

    Parâmetros
    ----------
    path : Path
        Caminho para o arquivo .h5ad.
    label : str
        Rótulo descritivo (ex.: "E8.5") usado nas mensagens.
    sample_cells : int
        Se > 0, amostra aleatoriamente este número de células.

    Retorna
    -------
    ad.AnnData
        O dataset carregado (e possivelmente amostrado).
    """
    print(f"\n📂  Carregando {label}: {path}")
    adata = sc.read_h5ad(path)
    print(f"    {adata.n_obs:,} células × {adata.n_vars:,} genes")

    if sample_cells and sample_cells < adata.n_obs:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=sample_cells, replace=False)
        adata = adata[idx].copy()
        print(f"    ↳ Amostrado para {adata.n_obs:,} células")

    return adata


# ─── Normalização ─────────────────────────────────────────────────────────────

def check_and_normalize(adata: ad.AnnData, label: str = "dataset") -> ad.AnnData:
    """
    Garante que o AnnData está em escala log1p-normalizada antes de qualquer
    análise de variância ou correlação.

    A heurística de detecção é conservadora: se o valor máximo da matriz for
    <= 20, consideramos que os dados já estão em log1p (onde expressões altas
    raramente ultrapassam 10–12). Caso contrário, aplica:
        1. normalize_total(target_sum=1e4)  → corrige library size
        2. log1p()                          → comprime a escala

    Por que isso importa
    --------------------
    - Em contagens brutas a variância é proporcional à expressão média
      (relação de Poisson/NB), então um gene muito expresso terá variância
      alta independentemente de variar biologicamente.
    - O threshold de variância só tem sentido interpretável em log-espaço,
      onde valores tipicamente ficam em [0, 10] e a variância de um gene
      completamente silencioso é 0 enquanto a de um gene diferencial gira
      em torno de 0.1–2.

    Parâmetros
    ----------
    adata : ad.AnnData
        Dataset de entrada (modificado in-place se normalização for aplicada).
    label : str
        Rótulo para as mensagens de log.

    Retorna
    -------
    ad.AnnData
        Dataset garantidamente normalizado.
    """
    print(f"\n── Verificação de normalização [{label}] ──────────────────────")
    X = adata.X
    raw_max = float(X.max() if not issparse(X) else X.data.max()) if (
        not issparse(X) or X.nnz > 0
    ) else 0.0

    print(f"  Valor máximo na matriz : {raw_max:.4f}")

    if raw_max <= 20.0:
        print("  Dados já em escala log1p — nenhuma normalização aplicada.")
        return adata

    print("  Dados parecem ser contagens brutas (max > 20).")
    print("  Aplicando: normalize_total(target_sum=1e4) + log1p …")
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    new_max = float(adata.X.max() if not issparse(adata.X) else adata.X.data.max())
    print(f"  Novo valor máximo      : {new_max:.4f}  (esperado <= ~15)")
    print("  Normalização concluida.")
    return adata


def _dense_X(adata: ad.AnnData) -> np.ndarray:
    """Retorna a matriz de expressão como np.ndarray float32 denso."""
    X = adata.X
    if issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# ─── 1. Filtro de baixa variância ─────────────────────────────────────────────

def filter_low_variance(
    adata: ad.AnnData,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    label: str = "dataset",
) -> ad.AnnData:
    """
    Remove genes cuja variância (entre células) seja inferior ao limiar dado.

    A tabela de genes removidos é salva em:
        <output_dir>/<label>_removed_genes.csv

    Colunas da tabela:
        gene_name   – nome do gene
        variance    – variância calculada
        mean_expr   – expressão média
        pct_nonzero – fração de células com expressão > 0

    Parâmetros
    ----------
    adata : ad.AnnData
        Dataset de entrada.
    threshold : float
        Variância mínima para manter o gene.
    output_dir : Path
        Pasta de saída para as tabelas CSV.
    label : str
        Prefixo usado no nome do arquivo de saída.

    Retorna
    -------
    ad.AnnData
        Novo AnnData contendo apenas os genes que passaram no filtro.
    """
    print(f"\n── Filtro de baixa variância (threshold={threshold}) ─────────────")
    t0 = time.perf_counter()

    X = _dense_X(adata)

    variances = X.var(axis=0)        # variância entre células (por gene)
    means = X.mean(axis=0)
    pct_nonzero = (X > 0).mean(axis=0)

    mask_keep = variances >= threshold
    n_removed = int((~mask_keep).sum())
    n_kept = int(mask_keep.sum())

    print(f"  Genes originais  : {adata.n_vars:,}")
    print(f"  Threshold        : {threshold}")
    print(f"  Genes removidos  : {n_removed:,} ({100 * n_removed / adata.n_vars:.1f}%)")
    print(f"  Genes mantidos   : {n_kept:,} ({100 * n_kept / adata.n_vars:.1f}%)")

    # Salva tabela de removidos
    removed_df = pd.DataFrame({
        "gene_name":   adata.var_names[~mask_keep].tolist(),
        "variance":    variances[~mask_keep],
        "mean_expr":   means[~mask_keep],
        "pct_nonzero": pct_nonzero[~mask_keep],
    }).sort_values("variance", ascending=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    removed_path = output_dir / f"{label}_removed_genes.csv"
    removed_df.to_csv(removed_path, index=False)
    print(f"  💾  Tabela de removidos → {removed_path}")

    # Estatísticas rápidas da tabela de removidos
    if n_removed > 0:
        zero_genes = int((variances == 0).sum())
        print(f"  └─ {zero_genes:,} genes com variância == 0 (completamente silenciosos)")
        print(f"     Variância mínima entre removidos: {removed_df['variance'].min():.6f}")
        print(f"     Variância máxima entre removidos: {removed_df['variance'].max():.6f}")

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  Concluído em {elapsed:.1f}s")

    return adata[:, mask_keep].copy()


# ─── 2. Agrupamento de genes correlacionados ──────────────────────────────────

def group_correlated_genes(
    adata: ad.AnnData,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    label: str = "dataset",
    chunk_size: int = 2_000,
) -> dict[str, list[str]]:
    """
    Agrupa genes com correlação de Pearson >= threshold usando uma estratégia
    de "union-find" simples: varre pares de genes por blocos (chunks) para
    não explodir a memória, e une genes que superam o limiar de correlação.

    A tabela de grupos é salva em:
        <output_dir>/<label>_gene_groups.csv

    Colunas da tabela:
        group_id          – identificador numérico do grupo (grupos de 1 gene = singleton)
        representative    – gene escolhido como representante do grupo (maior média)
        members           – lista separada por "|" de todos os membros
        group_size        – número de genes no grupo
        min_correlation   – correlação de Pearson mínima observada dentro do grupo
        max_correlation   – correlação de Pearson máxima observada dentro do grupo
        mean_expr_rep     – expressão média do representante

    Parâmetros
    ----------
    adata : ad.AnnData
        Dataset filtrado (saída de filter_low_variance).
    threshold : float
        Limiar de correlação de Pearson para considerar dois genes "correlacionados".
    output_dir : Path
        Pasta de saída para as tabelas CSV.
    label : str
        Prefixo usado no nome do arquivo de saída.
    chunk_size : int
        Número de genes processados por bloco (trade-off memória × velocidade).

    Retorna
    -------
    dict[str, list[str]]
        Dicionário  {representante → [membros]} para grupos com >= 2 genes.
        Genes singleton não aparecem (não formaram nenhum par correlacionado).
    """
    print(f"\n── Agrupamento de genes correlacionados (threshold={threshold}) ──")
    t0 = time.perf_counter()

    gene_names = np.array(adata.var_names)
    n_genes = len(gene_names)
    X = _dense_X(adata)  # (células × genes)

    # Normaliza colunas (z-score) para correlação de Pearson eficiente via produto matricial
    print(f"  Normalizando {n_genes:,} genes para cálculo de correlação…")
    col_mean = X.mean(axis=0)
    col_std  = X.std(axis=0)
    # Evita divisão por zero (genes com std == 0 não devem existir após filtro de variância,
    # mas é uma salvaguarda)
    col_std[col_std == 0] = 1.0
    X_z = (X - col_mean) / col_std  # (células × genes), float32

    n_cells = X_z.shape[0]

    # Union-Find ─────────────────────────────────────────────────────────────
    parent = list(range(n_genes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Rastreia correlações observadas por par de raízes para estatísticas
    # corr_records: list of (i_orig, j_orig, corr_value)
    corr_records: list[tuple[int, int, float]] = []

    # Varre blocos para calcular correlação sem materializar a matriz n_genes × n_genes inteira
    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    total_pairs_checked = 0

    print(f"  Varrendo pares em {n_chunks} blocos de até {chunk_size:,} genes…")

    for ci in range(n_chunks):
        i_start, i_end = ci * chunk_size, min((ci + 1) * chunk_size, n_genes)
        Xi = X_z[:, i_start:i_end]  # (cells × chunk_i)

        for cj in range(ci, n_chunks):
            j_start, j_end = cj * chunk_size, min((cj + 1) * chunk_size, n_genes)
            Xj = X_z[:, j_start:j_end]  # (cells × chunk_j)

            # Correlação de Pearson = dot(X_z_i, X_z_j) / n_cells
            corr_block = (Xi.T @ Xj) / n_cells  # (chunk_i × chunk_j)

            # Pares dentro do mesmo bloco: evitar diagonal (gene com si mesmo)
            pairs_i, pairs_j = np.where(corr_block >= threshold)

            for pi, pj in zip(pairs_i, pairs_j):
                gi = i_start + int(pi)
                gj = j_start + int(pj)
                if gi == gj:
                    continue  # pula diagonal
                if gi > gj and ci == cj:
                    continue  # evita duplicatas no mesmo bloco

                corr_val = float(corr_block[pi, pj])
                union(gi, gj)
                corr_records.append((gi, gj, corr_val))
                total_pairs_checked += 1

        # Progresso a cada 5 blocos ou no último
        if ci % 5 == 0 or ci == n_chunks - 1:
            elapsed = time.perf_counter() - t0
            print(f"    bloco {ci+1:>4}/{n_chunks} | pares correlacionados até agora: "
                  f"{total_pairs_checked:,} | {elapsed:.1f}s")

    # Monta grupos a partir do Union-Find ────────────────────────────────────
    from collections import defaultdict
    root_to_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n_genes):
        root_to_members[find(i)].append(i)

    # Estatísticas de correlação por grupo (min/max)
    # Para grupos de 2+ genes: procura os registros associados
    # Constrói mapa: gene_idx → root
    gene_to_root = {i: find(i) for i in range(n_genes)}

    # Agrupa corr_records por raiz
    root_corr: dict[int, list[float]] = defaultdict(list)
    for gi, gj, cv in corr_records:
        root = gene_to_root[gi]
        root_corr[root].append(cv)

    # Calcula expressão média por gene (para escolher representante)
    gene_means = X.mean(axis=0)

    # Monta DataFrame de grupos ───────────────────────────────────────────────
    rows = []
    group_id = 0
    gene_groups: dict[str, list[str]] = {}

    for root, members in root_to_members.items():
        size = len(members)
        member_names = gene_names[members].tolist()

        # Escolhe representante = gene com maior expressão média no grupo
        rep_idx_local = int(np.argmax(gene_means[members]))
        representative = member_names[rep_idx_local]

        corrs = root_corr.get(root, [])
        min_corr = float(np.min(corrs)) if corrs else float("nan")
        max_corr = float(np.max(corrs)) if corrs else float("nan")

        rows.append({
            "group_id":        group_id,
            "representative":  representative,
            "members":         "|".join(member_names),
            "group_size":      size,
            "min_correlation": min_corr,
            "max_correlation": max_corr,
            "mean_expr_rep":   float(gene_means[members[rep_idx_local]]),
        })

        if size >= 2:
            gene_groups[representative] = member_names

        group_id += 1

    groups_df = pd.DataFrame(rows).sort_values("group_size", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / f"{label}_gene_groups.csv"
    groups_df.to_csv(groups_path, index=False)
    print(f"\n  💾  Tabela de grupos → {groups_path}")

    # Resumo ──────────────────────────────────────────────────────────────────
    n_multi = int((groups_df["group_size"] >= 2).sum())
    n_singletons = int((groups_df["group_size"] == 1).sum())
    n_genes_grouped = int(groups_df.loc[groups_df["group_size"] >= 2, "group_size"].sum())

    print(f"\n  Resumo dos grupos:")
    print(f"    Grupos com >= 2 genes : {n_multi:,}")
    print(f"    Genes agrupados       : {n_genes_grouped:,} "
          f"({100 * n_genes_grouped / n_genes:.1f}% do total)")
    print(f"    Genes singleton       : {n_singletons:,}")
    if n_multi > 0:
        top5 = groups_df[groups_df["group_size"] >= 2].head(5)
        print(f"\n  Top-5 maiores grupos:")
        for _, row in top5.iterrows():
            members_preview = row["members"].split("|")[:5]
            dots = "…" if row["group_size"] > 5 else ""
            print(f"    [{int(row['group_size'])} genes] rep={row['representative']} "
                  f"corr=[{row['min_correlation']:.3f}, {row['max_correlation']:.3f}] "
                  f"membros={', '.join(members_preview)}{dots}")

    elapsed = time.perf_counter() - t0
    print(f"\n  ⏱  Concluído em {elapsed:.1f}s")

    return gene_groups


# ─── Relatório final ──────────────────────────────────────────────────────────

def print_final_report(
    original_n_genes: int,
    filtered: ad.AnnData,
    gene_groups: dict[str, list[str]],
) -> None:
    """Imprime um resumo consolidado do pré-processamento."""
    n_kept = filtered.n_vars
    n_multi_genes = sum(len(v) for v in gene_groups.values())
    n_representatives = len(gene_groups)

    print("\n" + "=" * 60)
    print("  RELATÓRIO FINAL DO PRÉ-PROCESSAMENTO")
    print("=" * 60)
    print(f"  Genes originais           : {original_n_genes:,}")
    print(f"  Genes após filtro de var  : {n_kept:,}  "
          f"(-{original_n_genes - n_kept:,})")
    print(f"  Grupos correlacionados    : {n_representatives:,} grupos "
          f"({n_multi_genes:,} genes → {n_representatives:,} representantes)")
    print(f"  Genes únicos efetivos     : {n_kept - n_multi_genes + n_representatives:,}  "
          f"(após colapsar grupos)")
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pré-processamento de expressão gênica – Virtual Embryo Challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--e85", type=Path, default=Path("data/E85.h5ad"),
        help="Caminho para o dataset E8.5",
    )
    parser.add_argument(
        "--e95", type=Path, default=Path("data/E95.h5ad"),
        help="Caminho para o dataset E9.5",
    )
    parser.add_argument(
        "--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
        help="Variância mínima para manter um gene",
    )
    parser.add_argument(
        "--correlation-threshold", type=float, default=DEFAULT_CORRELATION_THRESHOLD,
        help="Limiar de correlação de Pearson para agrupar genes",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Pasta para salvar as tabelas CSV de saída",
    )
    parser.add_argument(
        "--sample-cells", type=int, default=DEFAULT_SAMPLE_CELLS,
        help="Número de células a amostrar por dataset (0 = usar todas)",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PRÉ-PROCESSAMENTO – Virtual Embryo Challenge      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  variance_threshold    = {args.variance_threshold}")
    print(f"  correlation_threshold = {args.correlation_threshold}")
    print(f"  output_dir            = {args.output_dir}")
    print(f"  sample_cells          = {args.sample_cells if args.sample_cells else 'todos'}")

    # ── Carrega os dois datasets ──────────────────────────────────────────────
    adata_e85 = load_dataset(args.e85, label="E8.5", sample_cells=args.sample_cells)
    adata_e95 = load_dataset(args.e95, label="E9.5", sample_cells=args.sample_cells)

    # ── Normalização (deve vir ANTES de qualquer análise de variância) ────────
    # Sem isso, variâncias em contagens brutas chegam a milhares e o threshold
    # de 0.01 perde completamente o sentido.
    adata_e85 = check_and_normalize(adata_e85, label="E8.5")
    adata_e95 = check_and_normalize(adata_e95, label="E9.5")

    # Verifica se compartilham os mesmos genes (obrigatório para concatenar)
    assert list(adata_e85.var_names) == list(adata_e95.var_names), (
        "Os dois datasets não compartilham a mesma lista de genes!"
    )

    original_n_genes = adata_e85.n_vars

    # ── Concatena para calcular variância e correlação em conjunto ────────────
    print("\n🔗  Concatenando E8.5 + E9.5 para análise conjunta…")

    adata_joint = ad.concat([adata_e85, adata_e95], axis=0, label="stage",
                             keys=["E85", "E95"])
    print(f"    Total: {adata_joint.n_obs:,} células × {adata_joint.n_vars:,} genes")
    print(f"    (variâncias agora calculadas em log1p-space — threshold={args.variance_threshold} é interpretável)")


    # ── 1. Filtro de baixa variância ─────────────────────────────────────────
    adata_filtered = filter_low_variance(
        adata_e85,
        # adata_joint,
        threshold=args.variance_threshold,
        output_dir=args.output_dir,
        label="joint",
    )

    # ── 2. Agrupamento de correlacionados ─────────────────────────────────────
    gene_groups = group_correlated_genes(
        adata_filtered,
        threshold=args.correlation_threshold,
        output_dir=args.output_dir,
        label="joint",
    )

    # ── Relatório final ───────────────────────────────────────────────────────
    print_final_report(original_n_genes, adata_filtered, gene_groups)

    print(f"\n✅  Tudo pronto! Tabelas salvas em: {args.output_dir.resolve()}\n")
