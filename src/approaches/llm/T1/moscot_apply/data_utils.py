"""Utilidades para carregar e preparar E8.5/E9.5 antes do Optimal Transport.

Pontos importantes:
- Os dois AnnData são concatenados mantendo apenas os genes em comum (deveria ser
  a totalidade dos ~12.930 genes já filtrados, se o filtro de variância mínima foi
  aplicado de forma idêntica aos dois arquivos).
- A PCA usada como espaço de custo do OT é calculada de forma CONJUNTA (E8.5 + E9.5
  juntos), e não célula-a-célula separadamente por timepoint. Isso é essencial: se
  cada timepoint tivesse sua própria PCA, os dois espaços não seriam comparáveis e
  a distância usada pelo OT perderia sentido.
- Nenhuma etapa aqui usa obs['celltype'] para filtrar, ponderar ou restringir nada:
  a coluna é apenas preservada como metadado para inspeção posterior dos pares.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import scanpy as sc

logger = logging.getLogger(__name__)


def load_timepoints(
    e85_path: Union[str, Path],
    e95_path: Union[str, Path],
    time_e85: float = 8.5,
    time_e95: float = 9.5,
) -> ad.AnnData:
    """Carrega E8.5 e E9.5, restringe à interseção de genes e concatena.

    Adiciona:
        obs['time']       -> float, exigido pelo moscot.TemporalProblem (time_key)
        obs['timepoint']  -> string legível ('E8.5' / 'E9.5')
        obs['orig_index'] -> barcode original da célula (antes do prefixo)
        obs['batch']      -> rótulo de origem ('E8.5' / 'E9.5')

    obs_names recebem prefixo (E85_/E95_) para evitar colisão de barcodes entre
    os dois arquivos ao concatenar.
    """
    e85_path, e95_path = Path(e85_path), Path(e95_path)
    logger.info("Lendo %s", e85_path)
    a85 = sc.read_h5ad(e85_path)
    logger.info("Lendo %s", e95_path)
    a95 = sc.read_h5ad(e95_path)

    common_genes = a85.var_names.intersection(a95.var_names)
    if len(common_genes) < min(a85.n_vars, a95.n_vars):
        logger.warning(
            "Genes em comum entre E8.5 e E9.5: %d (E8.5 tinha %d, E9.5 tinha %d). "
            "Mantendo apenas a interseção.",
            len(common_genes), a85.n_vars, a95.n_vars,
        )
    else:
        logger.info("Genes em comum: %d (conjuntos idênticos).", len(common_genes))

    a85 = a85[:, common_genes].copy()
    a95 = a95[:, common_genes].copy()

    a85.obs["time"] = float(time_e85)
    a95.obs["time"] = float(time_e95)
    a85.obs["timepoint"] = f"E{time_e85}"
    a95.obs["timepoint"] = f"E{time_e95}"

    a85.obs["orig_index"] = a85.obs_names.astype(str)
    a95.obs["orig_index"] = a95.obs_names.astype(str)
    a85.obs_names = [f"E85_{n}" for n in a85.obs_names]
    a95.obs_names = [f"E95_{n}" for n in a95.obs_names]

    adata = ad.concat(
        [a85, a95],
        join="inner",
        label="batch",
        keys=[f"E{time_e85}", f"E{time_e95}"],
        index_unique=None,
    )
    adata.obs["time"] = adata.obs["time"].astype(float)
    logger.info(
        "Dataset concatenado: %d células (%d em E%.1f, %d em E%.1f), %d genes.",
        adata.n_obs, a85.n_obs, time_e85, a95.n_obs, time_e95, adata.n_vars,
    )
    return adata


def normalize_and_log(adata: ad.AnnData, target_sum: Optional[float] = 1e4) -> None:
    """Normalização padrão (total-count + log1p), em-place.

    Use --skip-normalization no CLI se os dados de entrada já estiverem
    normalizados/log-transformados (por exemplo, se o filtro de variância mínima
    já foi aplicado sobre dados log-normalizados).
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def compute_joint_pca(
    adata: ad.AnnData,
    n_comps: int = 50,
    use_hvg: bool = True,
    n_top_genes: int = 2000,
    batch_key: str = "batch",
) -> None:
    """Calcula uma PCA conjunta (mesmo espaço de coordenadas para E8.5 e E9.5) e
    salva o resultado em adata.obsm['X_pca_joint'], SEM alterar adata.X.

    Essa PCA é usada apenas como espaço de custo do OT (`joint_attr` do moscot) -
    a matriz de expressão original é preservada em adata.X/adata.layers para uso
    posterior no fine-tuning.
    """
    tmp = adata.copy()
    if use_hvg:
        sc.pp.highly_variable_genes(
            tmp, batch_key=batch_key, n_top_genes=min(n_top_genes, tmp.n_vars)
        )
        tmp = tmp[:, tmp.var["highly_variable"]].copy()
    sc.pp.scale(tmp, max_value=10)
    n_comps_eff = min(n_comps, tmp.n_vars - 1, tmp.n_obs - 1)
    sc.tl.pca(tmp, n_comps=n_comps_eff, svd_solver="arpack")
    adata.obsm["X_pca_joint"] = tmp.obsm["X_pca"]
    logger.info("PCA conjunta calculada: %d componentes (sobre %d genes).", n_comps_eff, tmp.n_vars)


def prepare_adata(
    e85_path: Union[str, Path],
    e95_path: Union[str, Path],
    skip_normalization: bool = False,
    recompute_pca: bool = True,
    n_pcs: int = 50,
    use_hvg: bool = True,
    n_top_genes: int = 2000,
) -> ad.AnnData:
    """Pipeline completo de preparação: carregar -> (normalizar) -> PCA conjunta."""
    adata = load_timepoints(e85_path, e95_path)

    if not skip_normalization:
        normalize_and_log(adata)
    else:
        logger.info("Pulando normalização (--skip-normalization).")

    if recompute_pca or "X_pca_joint" not in adata.obsm:
        compute_joint_pca(adata, n_comps=n_pcs, use_hvg=use_hvg, n_top_genes=n_top_genes)
    else:
        logger.info("Usando embedding já presente em adata.obsm['X_pca_joint'].")

    return adata