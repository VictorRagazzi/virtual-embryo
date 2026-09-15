"""
Preparação dos dados de treino: transforma expressão bruta em bins (entrada do
scGPT) e monta o pareamento probabilístico entre células de E8.5 e E9.5 a
partir do plano de transporte (Optimal Transport).
"""
import numpy as np
from scipy import sparse


def _to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def to_dense(matrix):
    """Versão pública de _to_dense, para uso fora deste arquivo."""
    return _to_dense(matrix)


def bin_expression_matrix(matrix, n_bins):
    """
    Reproduz o binning usado pelo scGPT: para cada célula, os valores de
    expressão diferentes de zero são divididos em n_bins-1 faixas (quantis).
    Valores zero continuam no bin 0.
    """
    matrix = _to_dense(matrix)
    binned = np.zeros_like(matrix, dtype=np.float32)

    for i in range(matrix.shape[0]):
        linha = matrix[i]
        mask = linha > 0
        if mask.sum() == 0:
            continue
        valores_validos = linha[mask]
        faixas = np.quantile(valores_validos, np.linspace(0, 1, n_bins - 1))
        binned[i, mask] = np.digitize(valores_validos, faixas)

    return binned


def _load_transport_plan(transport_path):
    """Abre o .npz do plano de transporte e devolve como matriz densa numpy."""
    try:
        plano = sparse.load_npz(transport_path)
        return plano.toarray()
    except ValueError:
        dados = np.load(transport_path, allow_pickle=True)
        chave = "transport" if "transport" in dados.files else dados.files[0]
        plano = dados[chave]
        return _to_dense(plano)


def get_best_partner_per_cell(transport_path):
    """
    Para cada célula de E8.5, retorna o índice da célula de E9.5 mais
    provável (maior peso na linha da matriz). Pareamento "duro" (1 só
    parceiro, sempre o mesmo) -- mantido aqui para comparação/depuração.
    """
    plano = _load_transport_plan(transport_path)
    return plano.argmax(axis=1)


def build_topk_partners(transport_path, k=5):
    """
    Para cada célula de E8.5 (linha do plano de transporte), guarda os k
    parceiros mais prováveis em E9.5 e seus pesos, normalizados para somar 1
    entre esses k parceiros.

    Isso permite sortear, a cada época de treino, um parceiro diferente
    (respeitando as probabilidades do OT) em vez de sempre usar o mesmo
    parceiro fixo (argmax) -- o que ajuda o modelo a ver mais variedade de
    pares e não "achatar" tanto a predição.
    """
    plano = _load_transport_plan(transport_path)
    n_celulas = plano.shape[0]

    topk_indices = np.zeros((n_celulas, k), dtype=np.int64)
    topk_pesos = np.zeros((n_celulas, k), dtype=np.float32)

    for i in range(n_celulas):
        linha = plano[i]
        maiores = np.argpartition(linha, -k)[-k:]
        pesos = linha[maiores].astype(np.float32)

        soma = pesos.sum()
        if soma <= 0:
            pesos = np.ones(k, dtype=np.float32) / k
        else:
            pesos = pesos / soma

        topk_indices[i] = maiores
        topk_pesos[i] = pesos

    return topk_indices, topk_pesos


def fill_missing_genes(adata_reference, gene_list, predicted_matrix):
    """
    Monta a matriz de expressão final com TODOS os genes do adata_reference.
    Para os genes que o scGPT realmente previu (gene_list), usamos a
    predição. Para os demais, copiamos o valor de E9.5.
    """
    x_final = _to_dense(adata_reference.X).copy()
    posicao_do_gene = {gene: i for i, gene in enumerate(adata_reference.var_names)}

    genes_nao_encontrados = [g for g in gene_list if g not in posicao_do_gene]
    if genes_nao_encontrados:
        raise ValueError(
            f"{len(genes_nao_encontrados)} genes previstos pelo modelo não existem "
            f"no adata de referência. Confira se é o mesmo E9.5 usado no treino."
        )

    for coluna_no_modelo, gene in enumerate(gene_list):
        coluna_final = posicao_do_gene[gene]
        x_final[:, coluna_final] = predicted_matrix[:, coluna_no_modelo]

    n_copiados = x_final.shape[1] - len(gene_list)
    print(f"[merge] {len(gene_list)} genes vieram do modelo, {n_copiados} genes copiados de E9.5.")
    return x_final


def build_training_arrays(adata85, adata95, transport_path, gene_list, n_bins):
    """Versão antiga (pareamento fixo por argmax). Mantida para compatibilidade."""
    adata85 = adata85[:, gene_list]
    adata95 = adata95[:, gene_list]

    parceiro_por_celula = get_best_partner_per_cell(transport_path)

    x_input = bin_expression_matrix(adata85.X, n_bins)
    x_target = _to_dense(adata95.X)[parceiro_por_celula]

    return x_input.astype(np.float32), x_target.astype(np.float32)


def build_training_arrays_from_paired(adata_paired, gene_list, n_bins):
    """Versão antiga (usa o .h5ad já pareado). Mantida para compatibilidade."""
    if "target" not in adata_paired.layers:
        raise ValueError("AnnData pareado precisa conter layers['target'] com os alvos E9.5.")

    adata_paired = adata_paired[:, gene_list]
    x_input = bin_expression_matrix(adata_paired.X, n_bins)
    x_target = _to_dense(adata_paired.layers["target"])

    return x_input.astype(np.float32), x_target.astype(np.float32)