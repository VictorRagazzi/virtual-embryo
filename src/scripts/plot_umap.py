"""
Reprodução do UMAP do Virtual Embryo Challenge — Task 1
========================================================

Pipeline: PCA (50 PCs) → UMAP
O arquivo original usou Harmony antes do UMAP, mas a coluna de batch
não está nos dados distribuídos — só celltype. Por isso rodamos sem Harmony,
que é o máximo que os dados disponíveis permitem.

Este script também compara o UMAP reproduzido com o UMAP oficial já
presente em adata.obsm['X_umap.harmony.rna'], lado a lado, e calcula
métricas de concordância entre os dois embeddings.

Uso:
    python umap_vec.py --h5ad T1_8.5.h5ad --out umap_8.5.png
    python umap_vec.py --h5ad T1_8.5.h5ad  # salva em umap_output.png
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from scipy.spatial import procrustes


# ─── parâmetros ────────────────────────────────────────────────────────────────

N_TOP_GENES   = 3000   # HVGs — padrão Scanpy/Seurat para scRNA-seq
N_PCS         = 50     # componentes PCA antes do UMAP
N_NEIGHBORS   = 30     # vizinhos para o grafo de células
MIN_DIST      = 0.3    # espalhamento do UMAP (maior = mais espalhado)
UMAP_METRIC   = "cosine"  # cosine é padrão quando se usa Harmony
RANDOM_STATE  = 42
REF_OBSM_KEY  = "X_umap.harmony.rna"   # UMAP oficial já presente nos dados
METRIC_SAMPLE = 5000                    # subamostra p/ métricas custosas (trustworthiness)


# ─── pipeline ──────────────────────────────────────────────────────────────────

def run(h5ad_path: str, out_path: str) -> None:

    # 1. Carregar
    print(f"Carregando {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} células × {adata.n_vars} genes")

    # 2. Guardar .X original (log1p-norm) antes de qualquer alteração
    adata.layers["logcounts"] = adata.X.copy()

    # 3. Seleção de genes altamente variáveis
    print(f"Selecionando {N_TOP_GENES} genes altamente variáveis ...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat_v3",
        layer="logcounts",
    )
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  {n_hvg} HVGs selecionados")

    # 4. PCA nos HVGs
    print(f"Calculando PCA ({N_PCS} componentes) ...")
    sc.tl.pca(
        adata,
        n_comps=N_PCS,
        use_highly_variable=True,
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )

    # 5. Grafo de vizinhos
    print(f"Construindo grafo de vizinhos (n_neighbors={N_NEIGHBORS}) ...")
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )

    # 6. UMAP
    print("Calculando UMAP ...")
    sc.tl.umap(
        adata,
        min_dist=MIN_DIST,
        random_state=RANDOM_STATE,
    )

    # 7. Pega o UMAP oficial, se existir
    ref_coords = adata.obsm.get(REF_OBSM_KEY)
    if ref_coords is None:
        print(f"[aviso] '{REF_OBSM_KEY}' não encontrado em adata.obsm — plotando só o meu.")
    else:
        ref_coords = np.asarray(ref_coords)

    # 8. Plot
    print(f"Gerando plot → {out_path}")
    _plot(adata, out_path, ref_coords)

    # 9. Métricas de comparação
    if ref_coords is not None:
        _compare(adata, ref_coords)

    print("Feito.")


# ─── visualização ──────────────────────────────────────────────────────────────

def _resolve_palette(celltypes):
    palette = None  # placeholder, resolvido em _plot
    return palette


def _scatter_embedding(ax, coords, celltypes, order, palette, extra_colors, title):
    for ct in order:
        mask = (celltypes == ct).values
        color = palette.get(ct, extra_colors.get(ct, "#aaaaaa"))
        if isinstance(color, dict):
            color = list(color.values())[0]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            s=3,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
    for ct in order:
        mask = (celltypes == ct).values
        if mask.sum() == 0:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.text(
            cx, cy, ct,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])


def _plot(adata: ad.AnnData, out_path: str, ref_coords=None) -> None:
    my_coords = adata.obsm["X_umap"]
    celltypes = adata.obs["celltype"].astype(str)
    palette   = adata.uns.get("celltype_palette", {})

    order = celltypes.value_counts().index.tolist()
    default_cmap = plt.cm.tab20.colors
    extra_colors = {}
    idx = 0
    for ct in order:
        if ct not in palette:
            extra_colors[ct] = default_cmap[idx % len(default_cmap)]
            idx += 1

    stage = h5ad_path.split("/")[-1].replace(".h5ad", "")

    if ref_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))
        _scatter_embedding(
            axes[0], my_coords, celltypes, order, palette, extra_colors,
            f"Reproduzido (PCA {N_PCS}PCs → UMAP, sem Harmony)\n{adata.n_obs} células",
        )
        _scatter_embedding(
            axes[1], ref_coords, celltypes, order, palette, extra_colors,
            f"Oficial ('{REF_OBSM_KEY}')\n{adata.n_obs} células",
        )
        fig.suptitle(f"Virtual Embryo — {stage}", fontsize=15)
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        _scatter_embedding(
            ax, my_coords, celltypes, order, palette, extra_colors,
            f"Virtual Embryo — {stage}\nPCA ({N_PCS} PCs) → UMAP | {adata.n_obs} células",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ─── métricas de comparação ──────────────────────────────────────────────────

def _compare(adata: ad.AnnData, ref_coords: np.ndarray) -> None:
    """
    Compara o embedding reproduzido com o oficial (mesmas células, mesma ordem).

    - Silhouette (por celltype): quão bem cada embedding separa os clusters
      biológicos conhecidos. É comparável entre os dois, não depende do PCA.
    - Trustworthiness: quão bem cada embedding 2D preserva a vizinhança local
      do espaço de PCA que eu calculei. É um proxy justo só para o MEU
      embedding — o oficial veio de um PCA com correção Harmony que não temos
      aqui — mas serve de referência de quão "distorcido" cada um ficou.
    - Procrustes: alinha os dois embeddings (rotação/escala/translação) e
      mede a dissimilaridade residual — quão parecidas são as duas formas.
    """
    my_coords = adata.obsm["X_umap"]
    x_pca     = adata.obsm["X_pca"]
    celltypes = adata.obs["celltype"].astype(str).values

    n = adata.n_obs
    if n > METRIC_SAMPLE:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(n, size=METRIC_SAMPLE, replace=False)
    else:
        sample_idx = np.arange(n)

    print("\n=== Métricas de comparação ===")

    sil_mine = silhouette_score(my_coords[sample_idx], celltypes[sample_idx])
    sil_ref  = silhouette_score(ref_coords[sample_idx], celltypes[sample_idx])
    print(f"Silhouette (por celltype)     — reproduzido: {sil_mine:.4f} | oficial: {sil_ref:.4f}")

    tw_mine = trustworthiness(x_pca[sample_idx], my_coords[sample_idx], n_neighbors=N_NEIGHBORS)
    tw_ref  = trustworthiness(x_pca[sample_idx], ref_coords[sample_idx], n_neighbors=N_NEIGHBORS)
    print(f"Trustworthiness vs meu PCA    — reproduzido: {tw_mine:.4f} | oficial: {tw_ref:.4f}")

    _, _, disparity = procrustes(my_coords[sample_idx], ref_coords[sample_idx])
    print(f"Procrustes disparity (reproduzido vs oficial): {disparity:.4f}  "
          f"(0 = formas idênticas após alinhamento; sem teto fixo — útil pra comparar entre rodadas)")


# ─── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, help="Caminho para o arquivo .h5ad")
    parser.add_argument("--out",  default="umap_output.png", help="Arquivo de saída (.png)")
    args = parser.parse_args()

    h5ad_path = args.h5ad
    run(args.h5ad, args.out)