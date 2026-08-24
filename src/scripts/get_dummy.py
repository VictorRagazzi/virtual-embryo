# import anndata as ad
# import numpy as np
# import scipy.sparse as sp

# # Carregar o dataset de treino só pra pegar os gene names na ordem certa
# import scanpy as sc
# adata_ref = sc.read_h5ad("data/E9.5_RNA.h5ad")  # ajuste o path

# N_CELLS = 2500
# gene_names = adata_ref.var_names  # 32.285 genes na ordem certa

# # Matriz toda zeros (sparse pra economizar memória)
# X_dummy = sp.csr_matrix((N_CELLS, len(gene_names)), dtype=np.float32)

# # Criar AnnData
# adata_pred = ad.AnnData(
#     X=X_dummy,
#     var=adata_ref.var.copy(),  # mantém o índice de genes na ordem certa
# )

# # Barcodes sintéticos
# adata_pred.obs_names = [f"SYNTH_{i:05d}" for i in range(N_CELLS)]

# adata_pred.write_h5ad("submission_dummy.h5ad")
# print(adata_pred)

import anndata as ad

adata = ad.read_h5ad("data/E85.h5ad")  # ajuste o nome/extensão real do arquivo
print("adata: ", adata)
print("adata.obs: ", adata.obs)
print("adata.uns: ", adata.uns)
print("adata.obsm: ", adata.obsm)
print("adata.layers: ", adata.layers)

suffixes = adata.obs_names.str.split("_").str[-1]
# print(suffixes.value_counts())

# Composição de tipos celulares em cada grupo
suffixes = adata.obs_names.str.split("_").str[-1]
adata.obs["batch"] = suffixes

import pandas as pd


# print(adata.obs["celltype"].value_counts())

# for celltype in adata.obs["celltype"].cat.categories:
#     print(f"{celltype}: {adata.obs[celltype].sum()}")