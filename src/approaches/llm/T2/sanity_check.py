"sanity_check.py"

import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from embed import embed_cells

# ─── Configuração ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PREDICTION_DIR = Path("predictions")
OUTPUT_PATH = PREDICTION_DIR / "prediction_e10_5_emb.h5ad"
N_HVG = 3000       # Genes altamente variáveis — reduz dimensionalidade
TOP_N_TEXT = 60    # Top genes por célula na serialização textual

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

e85, e95, hvgs, combined = load_and_preprocess()



df_e85 = pd.DataFrame(e85.X.toarray() if hasattr(e85.X, "toarray") else e85.X,
                        index=e85.obs_names, columns=hvgs)
df_e95 = pd.DataFrame(e95.X.toarray() if hasattr(e95.X, "toarray") else e95.X,
                        index=e95.obs_names, columns=hvgs)

emb_e85 = embed_cells(df_e85.sample(500), top_n=100)  # amostra pequena
emb_e95 = embed_cells(df_e95.sample(500), top_n=100)

all_emb = np.vstack([emb_e85, emb_e95])
labels = ["E8.5"] * 200 + ["E9.5"] * 200

# 1. Aumentando o PCA para 3 dimensões
pca = PCA(n_components=3).fit_transform(all_emb)

# 2. Configurando o ambiente 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Plotando com as 3 coordenadas (x, y, z)
ax.scatter(pca[:200, 0], pca[:200, 1], pca[:200, 2], label="E8.5", alpha=0.6)
ax.scatter(pca[200:, 0], pca[200:, 1], pca[200:, 2], label="E9.5", alpha=0.6)

# Opcional: Adicionando nomes aos eixos para ficar mais profissional
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')

ax.legend()
ax.set_title("LLM embedding space — E8.5 vs E9.5 (3D)")

# 4. Salvar ANTES do show!
plt.savefig("figs/sanity_check_llm_3d.png", bbox_inches='tight')
plt.show()