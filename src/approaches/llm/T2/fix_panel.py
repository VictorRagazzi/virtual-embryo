# fix_panel.py — corrige um arquivo já gerado
import anndata as ad
from align_panel import fetch_gene_panel, align_to_panel
from pathlib import Path

pred_path = Path("predictions/prediction_e10_5_emb.h5ad")
out_path  = Path("predictions/prediction_e10_5_emb_aligned.h5ad")

adata = ad.read_h5ad(pred_path)
panel_genes = fetch_gene_panel()
adata_fixed = align_to_panel(adata, panel_genes)
adata_fixed.write_h5ad(out_path)
print(f"Arquivo corrigido salvo em {out_path}")
print(f"Shape final: {adata_fixed.shape}")  # (n_cells, 32285)