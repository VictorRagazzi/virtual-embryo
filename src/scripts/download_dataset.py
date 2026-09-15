import scvelo as scv
import anndata as ad
import scanpy as sc
import numpy as np
import re

# =========================================================
# 1) CARREGAR REFERÊNCIA E DIAGNOSTICAR FORMATO
# =========================================================
ref = ad.read_h5ad("data/E85.h5ad")

print("=== REFERÊNCIA (E8.5) ===")
print("N genes:", ref.n_vars)
print("Exemplo de var_names:", ref.var_names[:5].tolist())
print("X dtype:", ref.X.dtype, "| max:", ref.X.max(), "| min:", ref.X.min())
is_integer_like = np.allclose(ref.X[:100].toarray() if hasattr(ref.X, "toarray") else ref.X[:100],
                               np.round(ref.X[:100].toarray() if hasattr(ref.X, "toarray") else ref.X[:100]))
print("Parece contagem bruta (inteiros)?", is_integer_like)
print("Parece já normalizado/log1p?", ref.X.max() < 20 and not is_integer_like)

ref_genes = ref.var_names.tolist()

# =========================================================
# 2) CARREGAR O ATLAS DE GASTRULAÇÃO
# =========================================================
adata = scv.datasets.gastrulation()

print("\n=== GASTRULAÇÃO (atlas) ===")
print("N genes:", adata.n_vars)
print("Exemplo de var_names:", adata.var_names[:5].tolist())

# checa sobreposição de nomes de genes direto
overlap = set(ref_genes) & set(adata.var_names)
print(f"Overlap direto de nomes de genes: {len(overlap)} / {len(ref_genes)}")

# se a sobreposição for baixa, provavelmente um lado usa ENSEMBL ID e outro usa símbolo
# nesse caso, tentamos usar adata.var (pode ter coluna com o outro identificador)
if len(overlap) < 0.5 * len(ref_genes):
    print("Overlap baixo — verificando colunas alternativas em adata.var:")
    print(adata.var.columns.tolist())
    print(adata.var.head())

# =========================================================
# 3) FILTRO AMPLIADO: linhagem cardíaca + precursores
# =========================================================
cardiac_lineage = [
    "Cardiomyocytes",
    "Pharyngeal mesoderm",
    "Nascent mesoderm",
    "Mixed mesoderm",
]
mask = adata.obs["celltype"].isin(cardiac_lineage)
adata_cardiac = adata[mask].copy()

print(f"\nCélulas na linhagem cardíaca ampliada: {adata_cardiac.n_obs}")
print(adata_cardiac.obs["celltype"].value_counts())
print(adata_cardiac.obs["stage"].value_counts())

# =========================================================
# 4) ALINHAR GENES AO ESPAÇO DA REFERÊNCIA
# =========================================================
# reindexa para ter EXATAMENTE os genes/ordem da referência;
# genes ausentes no atlas viram 0
adata_cardiac = adata_cardiac[:, [g for g in adata_cardiac.var_names if g in set(ref_genes)]].copy()

missing_genes = [g for g in ref_genes if g not in set(adata_cardiac.var_names)]
print(f"\nGenes da referência ausentes no atlas: {len(missing_genes)} de {len(ref_genes)}")

# cria matriz zerada no formato certo e preenche com o que existe
aligned = ad.AnnData(
    X=np.zeros((adata_cardiac.n_obs, len(ref_genes)), dtype=np.float32),
    obs=adata_cardiac.obs.copy(),
    var=ref.var.copy() if len(ref.var) == len(ref_genes) else None,
)
aligned.var_names = ref_genes

# preenche as colunas existentes
common = [g for g in ref_genes if g in adata_cardiac.var_names]
idx_ref = [ref_genes.index(g) for g in common]
sub = adata_cardiac[:, common]
X_sub = sub.X.toarray() if hasattr(sub.X, "toarray") else sub.X
aligned.X[:, idx_ref] = X_sub

# =========================================================
# 5) NORMALIZAR IGUAL À REFERÊNCIA (assume log1p(CP10k), ajuste se necessário)
# =========================================================
# ATENÇÃO: rode o diagnóstico do passo 1 antes de confiar nisso.
# Se a referência já está em log1p, replicamos o mesmo pipeline padrão:
sc.pp.normalize_total(aligned, target_sum=1e4)
sc.pp.log1p(aligned)

# =========================================================
# 6) SALVAR POR ESTÁGIO
# =========================================================
for stage in aligned.obs["stage"].unique():
    sub = aligned[aligned.obs["stage"] == stage].copy()
    stage_clean = re.sub(r'[^0-9A-Za-z]', '', stage)
    filename = f"{stage_clean}_ex.h5ad"
    sub.write(filename)
    print(f"Salvo: {filename}  ({sub.n_obs} células, {sub.n_vars} genes)")