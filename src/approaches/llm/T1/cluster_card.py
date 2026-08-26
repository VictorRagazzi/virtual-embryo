"""
Estagios 1-3 do pipeline LLM (Virtual Embryo Challenge - Task 1).

Como o dataset ja vem com `obs['celltype']` anotado, NAO fazemos clustering
do zero (Leiden/Louvain) -- usamos o rotulo existente como unidade de
"cluster" e computamos, para cada tipo celular:

  - proporcao da populacao em E8.5 e em E9.5
  - status: persistente (existe nos dois), novo em E9.5, ou perdido apos E8.5
  - top-N genes marcadores (via scanpy.tl.rank_genes_groups em E9.5,
    tipo vs. resto) com media de expressao em E8.5 e E9.5 e o delta

Saida: um JSON com um "card" por tipo celular -- e essa representacao
compacta que vai alimentar o prompt do LLM no Estagio 4, em vez do
vetor de expressao bruto (32k genes) por celula.

IMPORTANTE: este script assume que `.X` esta em uma escala consistente
entre E8.5 e E9.5 (ambos brutos OU ambos normalizados/log1p). Se `.X`
forem contagens brutas, use --normalize para aplicar normalize_total +
log1p antes de calcular medias e marcadores. Confira isso nos seus
arquivos reais antes de confiar nos deltas.
"""

import argparse
import json
import sys

import numpy as np
import scanpy as sc


def load_and_prep(path: str, normalize: bool) -> "sc.AnnData":
    adata = sc.read_h5ad(path)
    if "celltype" not in adata.obs.columns:
        sys.exit(f"{path}: obs['celltype'] nao encontrado")
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def mean_expr_per_gene(adata: "sc.AnnData", celltype: str, genes: list[str]) -> dict[str, float]:
    """Media de expressao de `genes` nas celulas de `celltype` (0.0 se o tipo nao existir)."""
    mask = adata.obs["celltype"] == celltype
    if mask.sum() == 0:
        return {g: 0.0 for g in genes}
    sub = adata[mask, genes]
    X = sub.X
    means = np.asarray(X.mean(axis=0)).ravel() if hasattr(X, "toarray") else X.mean(axis=0)
    return {g: float(m) for g, m in zip(genes, means)}


def build_cards(e85_path: str, e95_path: str, top_n: int, min_cells: int,
                 method: str, normalize: bool) -> list[dict]:
    e85 = load_and_prep(e85_path, normalize)
    e95 = load_and_prep(e95_path, normalize)

    types_e85 = set(e85.obs["celltype"].astype(str).unique())
    types_e95 = set(e95.obs["celltype"].astype(str).unique())

    n_e85, n_e95 = e85.n_obs, e95.n_obs

    counts_e85 = e85.obs["celltype"].astype(str).value_counts()
    counts_e95 = e95.obs["celltype"].astype(str).value_counts()

    # Marcadores calculados em E9.5 (tipo vs. resto) -- essa e' a "identidade atual"
    # de cada tipo celular que o LLM vai usar como base para extrapolar E10.5.
    valid_types_e95 = [t for t in types_e95 if counts_e95.get(t, 0) >= min_cells]
    e95_valid = e95[e95.obs["celltype"].astype(str).isin(valid_types_e95)].copy()
    sc.tl.rank_genes_groups(e95_valid, groupby="celltype", groups=valid_types_e95,
                             method=method, n_genes=top_n)

    cards = []
    all_types = sorted(types_e85 | types_e95)
    for ct in all_types:
        n85 = int(counts_e85.get(ct, 0))
        n95 = int(counts_e95.get(ct, 0))

        if ct in types_e85 and ct in types_e95:
            status = "persistent"
        elif ct in types_e95:
            status = "new_in_e9.5"
        else:
            status = "lost_after_e8.5"

        card = {
            "celltype": ct,
            "status": status,
            "n_cells_e8.5": n85,
            "n_cells_e9.5": n95,
            "proportion_e8.5": round(n85 / n_e85, 5) if n_e85 else 0.0,
            "proportion_e9.5": round(n95 / n_e95, 5) if n_e95 else 0.0,
            "top_markers": [],
        }

        if ct in valid_types_e95:
            genes = list(e95_valid.uns["rank_genes_groups"]["names"][ct])
            scores = list(e95_valid.uns["rank_genes_groups"]["scores"][ct])
            lfc = list(e95_valid.uns["rank_genes_groups"]["logfoldchanges"][ct])
            means_e95 = mean_expr_per_gene(e95, ct, genes)
            means_e85 = mean_expr_per_gene(e85, ct, genes) if ct in types_e85 else {g: None for g in genes}
            for g, sc_, lf in zip(genes, scores, lfc):
                m95 = means_e95[g]
                m85 = means_e85[g]
                card["top_markers"].append({
                    "gene": g,
                    "score": round(float(sc_), 3),
                    "log2fc_vs_rest_e9.5": round(float(lf), 3),
                    "mean_e8.5": round(m85, 4) if m85 is not None else None,
                    "mean_e9.5": round(m95, 4),
                    "delta_e8.5_to_e9.5": round(m95 - m85, 4) if m85 is not None else None,
                })

        cards.append(card)

    return cards


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e85", required=True)
    ap.add_argument("--e95", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--min-cells", type=int, default=20,
                     help="tipos com menos celulas que isso em E9.5 nao entram no rank_genes_groups")
    ap.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "t-test", "t-test_overestim_var"])
    ap.add_argument("--normalize", action="store_true",
                     help="aplica normalize_total + log1p antes de calcular medias/marcadores")
    args = ap.parse_args()

    cards = build_cards(args.e85, args.e95, args.top_n, args.min_cells, args.method, args.normalize)

    with open(args.out, "w") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)

    n_persist = sum(c["status"] == "persistent" for c in cards)
    n_new = sum(c["status"] == "new_in_e9.5" for c in cards)
    n_lost = sum(c["status"] == "lost_after_e8.5" for c in cards)
    print(f"{len(cards)} tipos celulares -> {args.out}")
    print(f"  persistentes: {n_persist} | novos em E9.5: {n_new} | perdidos apos E8.5: {n_lost}")


if __name__ == "__main__":
    main()