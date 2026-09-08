# import anndata as ad
# import numpy as np

# # =========================
# # Configurações
# # =========================

# INPUT_PATH = "data/E85.h5ad"
# OUTPUT_PATH = "predictions/E85_sample_1200.h5ad"

# N_CELLS = 1200
# SEED = 42  # para tornar a amostragem reproduzível


# # =========================
# # Carregar dataset
# # =========================

# adata = ad.read_h5ad(INPUT_PATH)

# print("Dataset original:")
# print(adata)


# # =========================
# # Amostrar células
# # =========================

# if N_CELLS > adata.n_obs:
#     raise ValueError(
#         f"N_CELLS ({N_CELLS}) é maior que o número de células "
#         f"disponíveis ({adata.n_obs})."
#     )

# rng = np.random.default_rng(SEED)

# indices = rng.choice(
#     adata.n_obs,
#     size=N_CELLS,
#     replace=False
# )

# adata_sample = adata[indices].copy()


# # =========================
# # Salvar
# # =========================

# adata_sample.write_h5ad(OUTPUT_PATH)

# print("\nDataset amostrado:")
# print(adata_sample)

# print(f"\nSalvo em: {OUTPUT_PATH}")




import anndata as ad

adata = ad.read_h5ad('data/E95.h5ad')
# print('adata: ', adata)
# print('adata.obs_names: ', adata.obs_names)
# print('adata.layers: ', adata.layers)
# print('adata.var: ', adata.var)
# print('adata.uns: ', adata.uns)


print(adata.X[:5,:5])
print(adata.X.min(), adata.X.max())