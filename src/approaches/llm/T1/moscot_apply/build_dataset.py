"""Monta o dataset final de pares (E8.5 -> E9.5) pronto para o fine-tuning
estilo perturb-GEP (scGPT): cada observação é um par (célula fonte, célula alvo).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def build_paired_anndata(
    adata_full: ad.AnnData,
    pairs: pd.DataFrame,
    layer: Optional[str] = None,
) -> ad.AnnData:
    """Constrói um AnnData "pareado" a partir da tabela de pares.

    - X                  : expressão da célula fonte (E8.5)  [n_pairs x n_genes]
    - layers['target']   : expressão da célula alvo (E9.5)   [n_pairs x n_genes]
    - obs                : metadados de origem/destino (celltype, índice original,
                            peso do OT, massa da linha) - só para inspeção/análise,
                            não usados para restringir o pareamento em si.
    """
    src_pos = adata_full.obs_names.get_indexer(pairs["source_cell"])
    tgt_pos = adata_full.obs_names.get_indexer(pairs["target_cell"])
    if (src_pos < 0).any() or (tgt_pos < 0).any():
        raise ValueError("Alguma célula em `pairs` não foi encontrada em adata_full.obs_names.")

    mat = adata_full.X if layer is None else adata_full.layers[layer]
    X_src = mat[src_pos]
    X_tgt = mat[tgt_pos]
    if sparse.issparse(X_src):
        X_src = X_src.copy()
        X_tgt = X_tgt.copy()

    has_celltype = "celltype" in adata_full.obs.columns
    obs = pd.DataFrame(
        {
            "source_cell": pairs["source_cell"].values,
            "target_cell": pairs["target_cell"].values,
            "ot_weight": pairs["weight"].values,
            "source_row_mass": pairs["row_mass"].values,
            "source_orig_index": adata_full.obs["orig_index"].values[src_pos],
            "target_orig_index": adata_full.obs["orig_index"].values[tgt_pos],
        }
    )
    if has_celltype:
        obs["source_celltype"] = adata_full.obs["celltype"].values[src_pos]
        obs["target_celltype"] = adata_full.obs["celltype"].values[tgt_pos]
        # apenas informativo: fração de pares em que o OT "trocou" o celltype
        changed = (obs["source_celltype"].astype(str) != obs["target_celltype"].astype(str)).mean()
        logger.info("Fração de pares com mudança de celltype (source != target): %.1f%%", 100 * changed)

    obs.index = pd.Index([f"pair_{i:07d}" for i in range(len(obs))], name="pair_id")

    paired = ad.AnnData(X=X_src, obs=obs, var=adata_full.var.copy())
    paired.layers["target"] = X_tgt
    paired.uns["pairing_info"] = {
        "n_pairs": int(len(obs)),
        "n_source_cells_used": int(pairs["source_cell"].nunique()),
        "n_target_cells_used": int(pairs["target_cell"].nunique()),
    }
    return paired


def save_outputs(
    paired: ad.AnnData,
    pairs: pd.DataFrame,
    out_dir: Union[str, Path],
    prefix: str = "e85_e95_ot_pairs",
) -> dict:
    """Salva o AnnData pareado (.h5ad) e a tabela de pares (.csv) em `out_dir`.

    Retorna um dict com os paths gerados.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = out_dir / f"{prefix}.h5ad"
    csv_path = out_dir / f"{prefix}.csv"

    paired.write_h5ad(h5ad_path)
    pairs.to_csv(csv_path, index=False)

    logger.info("Dataset pareado salvo em: %s (%d pares)", h5ad_path, paired.n_obs)
    logger.info("Tabela de pares (metadados) salva em: %s", csv_path)

    return {"h5ad": str(h5ad_path), "csv": str(csv_path)}