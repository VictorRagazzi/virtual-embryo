"""
visualize.py – Exploração e visualização de arquivos .h5ad do Virtual Embryo Challenge.

Uso:
    uv run python src/scripts/visualize.py --input data/<arquivo>.h5ad
    uv run python src/scripts/visualize.py --input data/<arquivo>.h5ad --output figs/
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


# ─── Config ───────────────────────────────────────────────────────────────────

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 5))


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> ad.AnnData:
    print(f"\n📂 Carregando: {path}")
    adata = sc.read_h5ad(path)
    print(adata)
    return adata


# ─── Sumário ──────────────────────────────────────────────────────────────────

def print_summary(adata: ad.AnnData) -> None:
    print("\n── Sumário ──────────────────────────────────────────")
    print(f"  Células   : {adata.n_obs:,}")
    print(f"  Genes     : {adata.n_vars:,}")
    print(f"  obs cols  : {list(adata.obs.columns)}")
    print(f"  var cols  : {list(adata.var.columns)}")
    print(f"  obsm keys : {list(adata.obsm.keys())}")
    print(f"  uns keys  : {list(adata.uns.keys())}")

    # Tipos celulares – tenta nomes comuns de coluna
    for col in ("cell_type", "celltype", "cluster", "leiden", "louvain", "annotation"):
        if col in adata.obs.columns:
            cts = adata.obs[col].value_counts()
            print(f"\n  '{col}' ({len(cts)} tipos):")
            print(cts.to_string())
            break


# ─── Preprocessamento mínimo ──────────────────────────────────────────────────

def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """
    Filtragem e normalização só se necessário.
    A competição já entrega log1p-normalizado, então apenas garantimos
    que HVGs e PCA/UMAP existam para visualizar.
    """
    adata = adata.copy()
    print("\n── Pré-processamento ────────────────────────────────")

    # Verifica se já está normalizado (valores raramente > 20 em log1p)
    raw_max = adata.X.max() if hasattr(adata.X, "max") else adata.X.data.max()
    print(f"  Valor máximo na matriz: {raw_max:.2f}")
    if raw_max > 50:
        print("  ⚠  Parece não normalizado – aplicando normalização + log1p")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("  ✓  Já normalizado (log1p detectado)")

    # HVGs
    if "highly_variable" not in adata.var.columns:
        print("  Selecionando HVGs...")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")

    # PCA
    if "X_pca" not in adata.obsm:
        print("  Computando PCA...")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=500)

    # Neighbors + UMAP
    if "X_umap" not in adata.obsm:
        print("  Computando neighbors + UMAP (pode demorar)...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)
    else:
        print("  ✓  UMAP já presente nos dados")

    return adata


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_cell_type_distribution(adata: ad.AnnData, out_dir: Path) -> None:
    """Barplot com proporção de cada tipo celular."""
    label_col = next(
        (c for c in ("cell_type", "celltype", "cluster", "leiden", "louvain", "annotation")
         if c in adata.obs.columns),
        None,
    )
    if label_col is None:
        print("  ⚠  Nenhuma coluna de tipo celular encontrada, pulando distribuição.")
        return

    counts = adata.obs[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.35)))
    counts.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Número de células")
    ax.set_title(f"Distribuição de tipos celulares ({label_col})")
    plt.tight_layout()
    save(fig, out_dir / "cell_type_distribution.png")


def plot_umap(adata: ad.AnnData, out_dir: Path) -> None:
    """UMAP colorido por tipo celular e por estágio (se disponível)."""
    label_col = next(
        (c for c in ("cell_type", "celltype", "cluster", "leiden", "louvain", "annotation")
         if c in adata.obs.columns),
        None,
    )

    colors = [c for c in [label_col, "stage", "time", "batch"] if c and c in adata.obs.columns]
    if not colors:
        colors = [None]  # só posição

    for color in colors:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color=color, ax=ax, show=False, frameon=False,
                   legend_loc="on data" if color else None)
        title = f"UMAP – {color}" if color else "UMAP"
        ax.set_title(title)
        fname = f"umap_{color or 'plain'}.png"
        save(fig, out_dir / fname)


def plot_top_genes(adata: ad.AnnData, out_dir: Path, n: int = 20) -> None:
    """Expressão média dos genes mais variáveis."""
    if "highly_variable" not in adata.var.columns:
        print("  ⚠  HVGs não encontrados, pulando plot de genes.")
        return

    hvg = adata.var[adata.var["highly_variable"]].index.tolist()[:n]
    # Média de expressão por gene
    if hasattr(adata.X, "toarray"):
        expr = np.asarray(adata[:, hvg].X.toarray()).mean(axis=0)
    else:
        expr = np.asarray(adata[:, hvg].X).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n), expr, color="coral", edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(hvg, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Expressão média (log1p)")
    ax.set_title(f"Top {n} genes altamente variáveis – média de expressão")
    plt.tight_layout()
    save(fig, out_dir / "top_hvg_expression.png")


def plot_pca_variance(adata: ad.AnnData, out_dir: Path) -> None:
    """Variância explicada por componente principal."""
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        print("  ⚠  Variância do PCA não encontrada, pulando.")
        return

    var_ratio = adata.uns["pca"]["variance_ratio"]
    cum = np.cumsum(var_ratio)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(var_ratio) + 1), var_ratio * 100, color="mediumseagreen", label="Por PC")
    ax.plot(range(1, len(cum) + 1), cum * 100, color="navy", linewidth=2, label="Acumulada")
    ax.axhline(90, color="red", linestyle="--", linewidth=0.8, label="90%")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Variância explicada (%)")
    ax.set_title("Variância explicada pelo PCA")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir / "pca_variance.png")

def print_expression_head(adata: ad.AnnData, n_cells: int = 10, n_genes: int = 8) -> None:
    import pandas as pd

    if hasattr(adata.X, "toarray"):
        matrix = adata.X[:n_cells].toarray()
    else:
        matrix = np.asarray(adata.X[:n_cells])

    genes = adata.var_names[:n_genes].tolist()
    cells = adata.obs_names[:n_cells].tolist()

    df = pd.DataFrame(matrix[:, :n_genes], index=cells, columns=genes)

    print(f"\n── Expressão gênica (primeiras {n_cells} células × {n_genes} genes) ───")
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\n  ... {adata.n_obs:,} células e {adata.n_vars:,} genes no total")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Salvo: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualização de .h5ad – Virtual Embryo Challenge")
    parser.add_argument("--input", required=True, type=Path, help="Caminho para o arquivo .h5ad")
    parser.add_argument("--output", type=Path, default=Path("figs"), help="Pasta de saída dos plots (default: figs/)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Pula pré-processamento (assume que UMAP já existe)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    adata = load(args.input)
    print_summary(adata)

    adata_pc = preprocess(adata) if not args.skip_preprocess else adata

    plot_cell_type_distribution(adata_pc, args.output)
    plot_umap(adata_pc, args.output)
    plot_pca_variance(adata_pc, args.output)
    plot_top_genes(adata_pc, args.output)
    print_expression_head(adata, n_cells=10, n_genes=8)

    print(f"\n✅ Pronto! Plots salvos em: {args.output.resolve()}")


if __name__ == "__main__":
    main()