# src/approaches/llm/align_panel.py

import numpy as np
import anndata as ad
import pandas as pd
import requests
from pathlib import Path

PANEL_URL = "https://virtualembryo.ai/challenge/panels/T1__val.genes.txt"
PANEL_CACHE = Path("data/T1_val_genes.txt")


def fetch_gene_panel(url: str = PANEL_URL, cache: Path = PANEL_CACHE) -> list[str]:
    """Baixa o painel oficial e retorna lista de genes na ordem correta."""
    if cache.exists():
        print(f"Usando painel cacheado: {cache}")
        return cache.read_text().strip().splitlines()

    print(f"Baixando painel de {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    print(f"Painel salvo: {len(r.text.splitlines())} genes")
    return r.text.strip().splitlines()


def align_to_panel(adata: ad.AnnData, panel_genes: list[str]) -> ad.AnnData:
    """
    Realinha um AnnData para o painel oficial.
    - Genes presentes na predição → mantém valores
    - Genes ausentes → preenche com 0.0
    - Ordem final = ordem exata do painel
    """
    pred_genes = set(adata.var_names)
    panel_set = set(panel_genes)

    n_present = len(pred_genes & panel_set)
    n_missing = len(panel_set - pred_genes)
    n_extra = len(pred_genes - panel_set)

    print(f"Genes no painel:     {len(panel_genes)}")
    print(f"Genes na predição:   {len(pred_genes)}")
    print(f"  → Presentes:       {n_present}")
    print(f"  → Ausentes (→ 0):  {n_missing}")
    print(f"  → Extras (drop):   {n_extra}")

    # Monta matriz final (n_cells × n_panel_genes) com zeros
    n_cells = adata.n_obs
    X_full = np.zeros((n_cells, len(panel_genes)), dtype=np.float32)

    # Preenche as colunas dos genes que temos
    pred_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    for i, gene in enumerate(panel_genes):
        if gene in pred_df.columns:
            X_full[:, i] = pred_df[gene].values

    # Monta o AnnData alinhado
    var_df = pd.DataFrame(index=panel_genes)
    var_df.index.name = "gene_id"

    adata_aligned = ad.AnnData(
        X=X_full,
        obs=adata.obs.copy(),
        var=var_df,
    )

    # Preserva embeddings e outros campos se existirem
    for key in adata.obsm:
        adata_aligned.obsm[key] = adata.obsm[key]

    return adata_aligned