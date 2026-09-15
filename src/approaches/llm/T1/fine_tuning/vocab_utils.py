"""
Utilitários para lidar com o vocabulário de genes do scGPT.

O vocabulário é um dicionário {nome_do_gene: id}. Precisamos dele porque o
scGPT trata cada gene como um "token", igual uma palavra em um modelo de
linguagem.
"""
import json


def load_vocab(vocab_path):
    with open(vocab_path) as f:
        return json.load(f)


def gene_to_vocab_id(gene, vocab):
    if gene in vocab:
        return vocab[gene]

    upper_gene = gene.upper()
    if upper_gene in vocab:
        return vocab[upper_gene]

    raise KeyError(f"Gene {gene!r} não encontrado no vocabulário do scGPT.")


def select_common_genes(adata, vocab, max_genes):
    """
    Seleção antiga: escolhe os genes por variância bruta. Mantida aqui só
    para referência/comparação -- o treino agora usa select_de_genes.
    """
    gene_names = adata.var_names.tolist()
    genes_in_vocab = [g for g in gene_names if g in vocab or g.upper() in vocab]

    if len(genes_in_vocab) == 0:
        raise ValueError(
            "Nenhum gene dos seus dados foi encontrado no vocabulário do scGPT. "
            "Confira se os nomes dos genes (var_names) usam o mesmo padrão do vocab.json."
        )

    if "variances" in adata.var.columns:
        variancias = adata.var.loc[genes_in_vocab, "variances"]
        genes_in_vocab = variancias.sort_values(ascending=False).index.tolist()

    genes_selecionados = genes_in_vocab[:max_genes]
    print(f"[vocab] {len(genes_selecionados)} genes selecionados por variância (de {len(gene_names)} totais).")
    return genes_selecionados


def select_de_genes(adata, vocab, max_genes, de_genes_path):
    """
    Escolhe os genes que mais mudam entre E8.5 e E9.5, na ordem calculada
    pelo script select_de_genes.py, filtrando só os que existem nos nossos
    dados (adata) e no vocabulário do scGPT.
    """
    with open(de_genes_path) as f:
        genes_ordenados_por_de = json.load(f)

    gene_names_disponiveis = set(adata.var_names)
    genes_validos = [
        g for g in genes_ordenados_por_de
        if g in gene_names_disponiveis and (g in vocab or g.upper() in vocab)
    ]

    if len(genes_validos) == 0:
        raise ValueError(
            "Nenhum gene da lista de expressão diferencial (de_genes_e85_e95.json) "
            "foi encontrado nos dados e no vocabulário do scGPT ao mesmo tempo. "
            "Rode select_de_genes.py de novo, ou confira os nomes dos genes."
        )

    genes_selecionados = genes_validos[:max_genes]
    print(
        f"[vocab] {len(genes_selecionados)} genes selecionados por diferença E8.5->E9.5 "
        f"(de {len(genes_ordenados_por_de)} candidatos rankeados)."
    )
    return genes_selecionados