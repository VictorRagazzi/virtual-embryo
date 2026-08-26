"""
Abordagem LLM para predição de E10.5.
Uso: python main.py
"""
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path

from align_panel import fetch_gene_panel, align_to_panel
from embed import embed_cells
from extrapolate import compute_cell_velocities, extrapolate_to_e105
from reconstruct import EmbeddingToExpressionDecoder

# ─── Configuração ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PREDICTION_DIR = Path("predictions")
OUTPUT_PATH = PREDICTION_DIR / "prediction_e10_5_emb.h5ad"
N_HVG = 3000       # Genes altamente variáveis — reduz dimensionalidade
TOP_N_TEXT = 60    # Top genes por célula na serialização textual
KNN_K = 5          # Vizinhos para cálculo de velocidade
N_OUTPUT_CELLS = 3000  # Células na predição final


def load_and_preprocess():
    """Carrega E8.5 e E9.5, normaliza, seleciona HVGs."""
    print("Carregando dados...")
    e85 = ad.read_h5ad(DATA_DIR / "E85.h5ad")
    e95 = ad.read_h5ad(DATA_DIR / "E95.h5ad")

    # Combinar para seleção conjunta de HVGs
    combined = ad.concat([e85, e95], label="timepoint", keys=["E8.5", "E9.5"])

    # Pipeline padrão scRNA-seq
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.highly_variable_genes(combined, n_top_genes=N_HVG, batch_key="timepoint")

    hvgs = combined.var_names[combined.var["highly_variable"]].tolist()
    print(f"HVGs selecionados: {len(hvgs)}")

    # Separar de volta
    e85_proc = combined[combined.obs["timepoint"] == "E8.5", hvgs]
    e95_proc = combined[combined.obs["timepoint"] == "E9.5", hvgs]

    return e85_proc, e95_proc, hvgs, combined


def main():
    e85, e95, hvgs, combined = load_and_preprocess()

    import pandas as pd
    df_e85 = pd.DataFrame(e85.X.toarray() if hasattr(e85.X, "toarray") else e85.X,
                           index=e85.obs_names, columns=hvgs)
    df_e95 = pd.DataFrame(e95.X.toarray() if hasattr(e95.X, "toarray") else e95.X,
                           index=e95.obs_names, columns=hvgs)

    # ── 1. Embedding ────────────────────────────────────────────────────────────
    print("\n[1/4] Gerando embeddings de E8.5...")
    emb_e85 = embed_cells(df_e85, top_n=TOP_N_TEXT)

    print("[1/4] Gerando embeddings de E9.5...")
    emb_e95 = embed_cells(df_e95, top_n=TOP_N_TEXT)

    # ── 2. Extrapolação temporal ─────────────────────────────────────────────────
    print("\n[2/4] Calculando velocidades celulares no espaço latente...")
    velocities = compute_cell_velocities(emb_e85, emb_e95, k=KNN_K)

    print("[2/4] Extrapolando para E10.5...")
    emb_e105_pred = extrapolate_to_e105(emb_e95, velocities, scale=1.0)

    # ── 3. Reconstrução da expressão gênica ─────────────────────────────────────
    print("\n[3/4] Treinando decoder embedding → expressão...")
    X_train = np.vstack([emb_e85, emb_e95])
    y_train = np.vstack([df_e85.values, df_e95.values])

    decoder = EmbeddingToExpressionDecoder(alpha=10.0)
    decoder.fit(X_train, y_train)

    print("[3/4] Reconstruindo expressão de E10.5...")
    expr_e105 = decoder.predict(emb_e105_pred)  # (n_e95, n_hvgs)

    # ── 4. Alinhar ao painel oficial e salvar ──────────────────────────────────────

    print("\n[4/4] Alinhando ao painel oficial...")
    panel_genes = fetch_gene_panel()

    # AnnData com apenas os HVGs (como estava antes)
    adata_pred_hvg = ad.AnnData(
        X=expr_e105[idx].astype(np.float32),
        obs=e95.obs.iloc[idx].copy(),
        var=pd.DataFrame(index=hvgs),
    )
    adata_pred_hvg.obs["timepoint"] = "E10.5"
    adata_pred_hvg.obsm["X_llm_emb"] = emb_e105_pred[idx]

    # Expande para os 32285 genes na ordem correta
    adata_final = align_to_panel(adata_pred_hvg, panel_genes)

    adata_final.write_h5ad(OUTPUT_PATH)
    print(f"Predição salva em {OUTPUT_PATH}")
    print(f"Shape: {adata_final.shape}")

    # ── Validação rápida ────────────────────────────────────────────────────────
    try:
        from veckit import score
        print("\nAvaliando com veckit (sanity check E9.5 vs E9.5)...")
        result = score(
            task="T1",
            input=str(OUTPUT_PATH),
            target=str(DATA_DIR / "E95.h5ad"),  # placeholder até ter E10.5
            reference=str(DATA_DIR / "E85.h5ad"),
        )
        print(result["metrics"])
    except Exception as e:
        print(f"Score não disponível: {e}")


if __name__ == "__main__":
    main()
