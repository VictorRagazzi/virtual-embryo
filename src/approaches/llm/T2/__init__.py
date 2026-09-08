"""
scgpt_decoder
=============

Pipeline pequeno para validar a ideia:

    célula (32k genes) --scGPT (encoder congelado)--> Y_512
    Y_512 --decoder (MLP treinado)--> reconstrução da célula (32k genes)

O objetivo NÃO é prever E10.5 de verdade: é checar se o embedding Y_512
do scGPT retém informação suficiente para reconstruir a expressão gênica
original, usando a métrica oficial do veckit como régua de sanidade sobre
o subconjunto de validação vindo só de E9.5 (conforme a documentação do
score, que espera target/reference reais).

Uso típico (ver main.py para a CLI completa):

    from scgpt_decoder import (
        get_or_compute_embeddings, stratified_split_indices,
        EmbeddingToExpressionDataset, train_decoder,
        ExpressionDecoder, decode_to_adata, run_veckit_score,
    )
"""

from .decoder import ExpressionDecoder
from .dataset import EmbeddingToExpressionDataset, stratified_split_indices
from .scgpt_embed import (
    compute_cell_embeddings,
    get_or_compute_embeddings,
    humanize_mouse_symbols,
)
from .train import train_decoder
from .evaluate import decode_to_adata, run_veckit_score

__all__ = [
    "ExpressionDecoder",
    "EmbeddingToExpressionDataset",
    "stratified_split_indices",
    "compute_cell_embeddings",
    "get_or_compute_embeddings",
    "humanize_mouse_symbols",
    "train_decoder",
    "decode_to_adata",
    "run_veckit_score",
]