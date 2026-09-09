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
import numpy as np

pred = ad.read_h5ad("data/prediction_e9_5_decoder_val.h5ad")
X = pred.X
print("shape:", X.shape)
print("variância média por gene:", X.var(axis=0).mean())
print("variância média entre as PRIMEIRAS 20 células (por linha, transposto):", X.var(axis=1).mean())
print("std entre células, gene 0:", X[:, 0].std(), "| valores gene 0, 5 primeiras células:", X[:5, 0])