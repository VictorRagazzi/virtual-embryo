import anndata as ad

aligned = ad.read_h5ad("data/E85_ex.h5ad")  # ou qualquer outro estágio
ref = ad.read_h5ad("data/E85.h5ad")

markers = ["Nkx2-5", "Tnnt2", "Myh6", "Gata4", "Tbx5", "Isl1"]
zeroed = [g for g in markers if aligned[:, g].X.sum() == 0]
print("Marcadores cardíacos zerados (ausentes no atlas):", zeroed)

# checa se a escala de normalização bate com a referência
print("Aligned X max:", aligned.X.max(), "| Ref X max:", ref.X.max())
print("Aligned X mean (não-zero):", aligned.X[aligned.X > 0].mean())
print("Ref X mean (não-zero):", ref.X[ref.X > 0].mean())