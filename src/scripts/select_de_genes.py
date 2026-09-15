"""
Rankeia os genes pela diferença de expressão entre E8.5 e E9.5 (em vez de
usar variância bruta, que pega genes tecnicamente ruidosos mas que não
necessariamente mudam entre os estágios).

A ideia: um gene que muda bastante de E8.5 para E9.5 tem mais chance de
continuar mudando (na mesma direção) em E10.5 -- e é exatamente esse tipo de
mudança que métricas como de_score/de_direction avaliam.

Rodar com:
    uv run python -m src.scripts.select_de_genes
"""
import json
from pathlib import Path

import anndata as ad
import scanpy as sc

E85_PATH = Path("data/E85.h5ad")
E95_PATH = Path("data/E95.h5ad")
OUTPUT_PATH = Path("data/de_genes_e85_e95.json")

# guardamos mais genes do que o necessário (o corte final para 1200, por
# causa do limite do scGPT, acontece na hora do treino)
N_TOP_GENES = 3000


def main():
    adata85 = sc.read_h5ad(E85_PATH)
    adata95 = sc.read_h5ad(E95_PATH)

    adata85.obs["estagio"] = "E8.5"
    adata95.obs["estagio"] = "E9.5"

    adata_concat = ad.concat([adata85, adata95], join="inner")
    print(f"[DE] {adata_concat.n_obs} células, {adata_concat.n_vars} genes em comum entre E8.5 e E9.5")

    sc.tl.rank_genes_groups(
        adata_concat,
        groupby="estagio",
        groups=["E9.5"],
        reference="E8.5",
        method="wilcoxon",
    )

    resultado = sc.get.rank_genes_groups_df(adata_concat, group="E9.5")
    resultado["abs_logfoldchange"] = resultado["logfoldchanges"].abs()
    resultado = resultado.sort_values("abs_logfoldchange", ascending=False)

    genes_ordenados = resultado["names"].tolist()[:N_TOP_GENES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(genes_ordenados, f)

    print(f"[DE] {len(genes_ordenados)} genes salvos em {OUTPUT_PATH}, ordenados por maior mudança E8.5->E9.5.")


if __name__ == "__main__":
    main()