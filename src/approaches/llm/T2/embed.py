"""
Converte células (vetores de expressão) em embeddings de linguagem.
Usa sentence-transformers ou API compatível.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# PubMedBERT-based — conhece vocabulário biológico
MODEL_NAME = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
# Fallback mais leve, ainda bom:
# MODEL_NAME = "all-mpnet-base-v2"


def cell_to_text(cell: pd.Series, top_n: int = 60) -> str:
    """
    Serializa um vetor de expressão como texto.
    Usa apenas os top_n genes mais expressos — o LLM conhece esses nomes.

    Sox17=3.2, Pou5f1=2.1 tem semântica biológica real no espaço latente.
    """
    expressed = cell[cell > 0].nlargest(top_n)
    if expressed.empty:
        return "Mouse embryo cardiac cell: no detected gene expression"

    genes_str = ", ".join(
        f"{gene}={val:.2f}" for gene, val in expressed.items()
    )
    return f"Mouse embryo cardiac cell gene expression: {genes_str}"


def embed_cells(
    expr_df: pd.DataFrame,
    top_n: int = 60,
    batch_size: int = 128,
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """
    Recebe DataFrame (células × genes), devolve matriz (células × embed_dim).
    """
    model = SentenceTransformer(model_name)

    texts = [cell_to_text(row, top_n=top_n) for _, row in expr_df.iterrows()]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity fica mais estável
    )
    return embeddings  # (n_cells, 768)