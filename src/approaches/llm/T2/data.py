"""Expressão, tokens e pares temporais. O pareamento pode ser trocado isoladamente."""

import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def pair_cells(source_expression, future_expression, strategy="nearest_neighbor"):
    """Retorna um índice e uma distância por origem; permite reutilizar destinos.

    As matrizes devem usar as mesmas coordenadas. Substitua somente esta função
    para experimentar outro pareamento. Estes são pseudo-pares, não linhagens.
    """
    if strategy != "nearest_neighbor":
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    if source_expression.shape[0] == 0 or future_expression.shape[0] == 0:
        raise ValueError("O pareamento exige células nos dois tempos.")
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute")
    neighbors.fit(future_expression)
    distances, indices = neighbors.kneighbors(source_expression)
    return indices[:, 0], distances[:, 0]


def load_dictionary(path):
    # Os arquivos pickle devem vir de uma fonte confiável, como o autor do modelo.
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare_expression(matrix, expression_scale):
    """Retorna expressão linear e log1p normalizado para 10 mil por célula.

    log1p pressupõe log natural sobre expressão linear, sem scaling/z-score.
    A normalização acontece antes de selecionar genes.
    """
    linear = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    linear.sum_duplicates()
    linear.eliminate_zeros()
    if not np.isfinite(linear.data).all() or (linear.data < 0).any():
        raise ValueError("Expressão deve ser finita e não negativa.")
    if expression_scale == "log1p":
        linear.data = np.expm1(linear.data)
    elif expression_scale == "counts":
        if not np.allclose(linear.data, np.round(linear.data), atol=1e-5, rtol=0):
            raise ValueError("Counts exige contagens inteiras; confirme a escala de X.")
    else:
        raise ValueError("Declare a escala: counts ou log1p.")
    totals = np.asarray(linear.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Células vazias ou expressão inválida após transformação.")
    normalized = linear.multiply((10000 / totals)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return linear, normalized


def tokenize_expression(linear, gene_ids, token_dictionary, gene_medians, max_length):
    """Rank de expressão/mediana, sem CLS/SEP, conforme o tokenizer Mouse-Geneformer."""
    if max_length < 1 or max_length > 2048:
        raise ValueError("O comprimento deve estar entre 1 e 2048.")
    columns = [i for i, gene in enumerate(gene_ids)
               if gene in token_dictionary and gene in gene_medians]
    if not columns:
        raise ValueError("Nenhum gene coincide com vocabulário e medianas. Verifique IDs.")
    matched = [gene_ids[i] for i in columns]
    if len(set(matched)) != len(matched):
        raise ValueError("IDs duplicados após mapeamento; resolva a ambiguidade antes do treino.")
    medians = np.asarray([gene_medians[g] for g in matched], dtype=np.float32)
    if not np.isfinite(medians).all() or (medians <= 0).any():
        raise ValueError("Medianas devem ser finitas e positivas.")
    tokens = np.asarray([token_dictionary[g] for g in matched], dtype=np.int64)
    pad_id = int(token_dictionary["<pad>"])
    if (tokens < 0).any() or (tokens == pad_id).any():
        raise ValueError("Token de gene inválido.")
    values = linear[:, columns].tocsr()
    input_ids = np.full((values.shape[0], max_length), pad_id, dtype=np.int64)
    mask = np.zeros_like(input_ids, dtype=bool)
    for row in range(values.shape[0]):
        start, end = values.indptr[row:row + 2]
        indices = values.indices[start:end]
        expression = values.data[start:end]
        positive = expression > 0
        indices, expression = indices[positive], expression[positive]
        if not len(indices):
            raise ValueError(f"Célula {row} sem genes expressos no vocabulário.")
        # O fator de biblioteca é constante dentro da célula e não muda o rank.
        order = np.argsort(-(expression / medians[indices]), kind="stable")[:max_length]
        ranked = tokens[indices[order]]
        input_ids[row, :len(ranked)] = ranked
        mask[row, :len(ranked)] = True
    return input_ids, mask


def read_stage(path, scale, token_dictionary, gene_medians, max_length, gene_map=None,
               layer=None, max_cells=None, seed=42, exclude_cells=()):
    data = ad.read_h5ad(path, backed="r")
    try:
        if not data.var_names.is_unique or not data.obs_names.is_unique:
            raise ValueError(f"Genes e células devem ter nomes únicos: {path}")
        indices = np.flatnonzero(~data.obs_names.isin(exclude_cells))
        if max_cells is not None and len(indices) > max_cells:
            indices = np.sort(np.random.default_rng(seed).choice(indices, max_cells, replace=False))
        # Leia somente a expressão solicitada, sem carregar obsm e outras layers.
        matrix = data.X if layer is None else data.layers[layer]
        subset = ad.AnnData(sparse.csr_matrix(matrix[indices, :], dtype=np.float32),
                            obs=data.obs.iloc[indices].copy(), var=data.var.copy())
    finally:
        data.file.close()
    genes = subset.var_names.astype(str).tolist()
    if "ensembl_id" in subset.var:
        gene_ids = subset.var["ensembl_id"].astype(str).tolist()
    elif gene_map is not None:
        gene_ids = [gene_map.get(gene, "") for gene in genes]
    else:
        gene_ids = genes
    gene_ids = [gene.split(".")[0] if gene.startswith("ENSMUSG") else gene for gene in gene_ids]
    linear, expression = prepare_expression(subset.X, scale)
    ids, mask = tokenize_expression(linear, gene_ids, token_dictionary, gene_medians, max_length)
    return {"expression": expression, "input_ids": ids, "mask": mask,
            "genes": genes, "obs": subset.obs.copy()}


def split_cells(observations, validation_fraction, seed, group_column=None):
    """Divide antes de parear; opcionalmente mantém embriões/lotes inteiros juntos."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1.")
    if group_column:
        groups = observations[group_column]
        if groups.isna().any():
            raise ValueError("Grupos de validação não podem estar ausentes.")
        units = groups.unique()
    else:
        units = np.arange(len(observations))
    if len(units) < 2:
        raise ValueError("Cada estágio precisa de pelo menos duas células/grupos.")
    units = np.random.default_rng(seed).permutation(units)
    count = min(len(units) - 1, max(1, int(len(units) * validation_fraction)))
    validation = observations[group_column].isin(units[:count]).to_numpy() if group_column else np.isin(np.arange(len(observations)), units[:count])
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def read_gene_map(path):
    if path is None:
        return None
    table = pd.read_csv(path, dtype=str)
    if table["gene_symbol"].duplicated().any() or table[["gene_symbol", "ensembl_id"]].isna().any().any():
        raise ValueError("Mapeamento exige símbolos únicos e IDs preenchidos.")
    return dict(zip(table.gene_symbol, table.ensembl_id))
