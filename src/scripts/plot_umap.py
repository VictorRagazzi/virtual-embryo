"""
Reprodução do UMAP do Virtual Embryo Challenge — Task 1 (com Harmony)
========================================================================

Pipeline: PCA (50 PCs) → Harmony (correção de batch) → UMAP

O batch é inferido do sufixo do nome das células (ex: '..._1', '..._2'),
que aparenta ser o rótulo de amostra/lote embutido no índice pelo
Scanpy/AnnData ao concatenar os dois objetos originais.

Uso:
    python umap_vec.py --h5ad T1_8.5.h5ad --out umap_8.5.png
    python umap_vec.py --h5ad T1_8.5.h5ad --no-harmony   # roda sem Harmony (versão anterior)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from scipy.spatial import procrustes


# ─── parâmetros ────────────────────────────────────────────────────────────────

N_TOP_GENES   = 3000
N_PCS         = 30
N_NEIGHBORS   = 30
MIN_DIST      = 0.3
UMAP_METRIC   = "cosine"
RANDOM_STATE  = 42
REF_OBSM_KEY  = "X_umap.harmony.rna"
METRIC_SAMPLE = 5000


# ─── pipeline ──────────────────────────────────────────────────────────────────

import pandas as pd
import harmonypy as hm

# ...

def _run_harmony(pca_coords: np.ndarray, batch: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Roda harmonypy diretamente, sem o wrapper do Scanpy — versões recentes
    (backend PyTorch) podem retornar Z_corr em formato diferente do que
    sce.pp.harmony_integrate espera."""
    meta_data = pd.DataFrame({"batch": batch})
    ho = hm.run_harmony(pca_coords, meta_data, ["batch"], random_state=random_state)

    z = ho.Z_corr
    if hasattr(z, "detach"):       # tensor torch
        z = z.detach().cpu().numpy()
    else:
        z = np.asarray(z)

    n_cells, n_pcs = pca_coords.shape
    if z.shape == (n_pcs, n_cells):
        z = z.T
    elif z.shape != (n_cells, n_pcs):
        raise ValueError(
            f"Formato inesperado de Z_corr: {z.shape} "
            f"(esperado ({n_cells}, {n_pcs}) ou ({n_pcs}, {n_cells}))"
        )
    return z

def run(h5ad_path: str, out_path: str, use_harmony: bool) -> None:

    print(f"Carregando {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} células × {adata.n_vars} genes")

    adata.layers["logcounts"] = adata.X.copy()

    print(f"Selecionando {N_TOP_GENES} genes altamente variáveis ...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3", layer="logcounts",
    )
    print(f"  {adata.var['highly_variable'].sum()} HVGs selecionados")

    print(f"Calculando PCA ({N_PCS} componentes) ...")
    sc.tl.pca(
        adata, n_comps=N_PCS, use_highly_variable=True,
        svd_solver="arpack", random_state=RANDOM_STATE,
    )

    pca_rep = "X_pca"

    if use_harmony:
        batch = _infer_batch(adata)
        if batch is None:
            print("[aviso] Não consegui inferir batch a partir do nome das células — seguindo sem Harmony.")
            use_harmony = False
        else:
            adata.obs["batch"] = batch
            print("Distribuição de batch inferida:")
            print(adata.obs["batch"].value_counts().to_string())
            print("Rodando Harmony (correção de batch sobre o PCA) ...")
            adata.obsm["X_pca_harmony"] = _run_harmony(
                adata.obsm["X_pca"], adata.obs["batch"].values, random_state=RANDOM_STATE,
            )
            pca_rep = "X_pca_harmony"

    print(f"Construindo grafo de vizinhos (n_neighbors={N_NEIGHBORS}, base='{pca_rep}') ...")
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep=pca_rep,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    
    print("Calculando UMAP ...")
    sc.tl.umap(adata, min_dist=MIN_DIST, random_state=RANDOM_STATE)

    ref_coords = adata.obsm.get(REF_OBSM_KEY)
    if ref_coords is None:
        print(f"[aviso] '{REF_OBSM_KEY}' não encontrado em adata.obsm — plotando só o meu.")
    else:
        ref_coords = np.asarray(ref_coords)

    print(f"Gerando plot → {out_path}")
    _plot(adata, out_path, ref_coords, use_harmony)

    if ref_coords is not None:
        _compare(adata, ref_coords, pca_rep)

    print("Feito.")


def _infer_batch(adata: ad.AnnData):
    """Extrai o sufixo após o último '_' no nome das células como rótulo de batch."""
    names = adata.obs_names.to_series()
    if not names.str.contains("_").all():
        return None
    suffix = names.str.rsplit("_", n=1).str[-1]
    # heurística: batch deve ter poucas categorias (não um ID único por célula)
    if suffix.nunique() < 1 or suffix.nunique() > 20:
        return None
    return suffix.values


# ─── visualização ──────────────────────────────────────────────────────────────

def _scatter_embedding(ax, coords, celltypes, order, palette, extra_colors, title):
    for ct in order:
        mask = (celltypes == ct).values
        color = palette.get(ct, extra_colors.get(ct, "#aaaaaa"))
        if isinstance(color, dict):
            color = list(color.values())[0]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], s=3, alpha=0.7, linewidths=0, rasterized=True,
        )
    for ct in order:
        mask = (celltypes == ct).values
        if mask.sum() == 0:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.text(
            cx, cy, ct, fontsize=10, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])


def _plot(adata: ad.AnnData, out_path: str, ref_coords, use_harmony: bool) -> None:
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
    my_label = "Reproduzido (PCA → Harmony → UMAP)" if use_harmony else "Reproduzido (PCA → UMAP, sem Harmony)"

    if ref_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))
        _scatter_embedding(
            axes[0], my_coords, celltypes, order, palette, extra_colors,
            f"{my_label}\n{adata.n_obs} células",
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
            f"Virtual Embryo — {stage}\n{my_label} | {adata.n_obs} células",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ─── métricas de comparação ──────────────────────────────────────────────────

def _compare(adata: ad.AnnData, ref_coords: np.ndarray, pca_rep: str) -> None:
    my_coords = adata.obsm["X_umap"]
    x_pca     = adata.obsm[pca_rep]
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
    print(f"Procrustes disparity (reproduzido vs oficial): {disparity:.4f}")

# ─── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, help="Caminho para o arquivo .h5ad")
    parser.add_argument("--out",  default="figs/umap_output.png", help="Arquivo de saída (.png)")
    parser.add_argument("--no-harmony", action="store_true", help="Roda sem Harmony (comportamento anterior)")
    args = parser.parse_args()

    h5ad_path = args.h5ad
    run(args.h5ad, args.out, use_harmony=not args.no_harmony)