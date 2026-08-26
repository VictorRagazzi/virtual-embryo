"""
Extrapolação temporal no espaço de embedding.

Estratégia principal: RNA Velocity–like no espaço latente.
Para cada célula em E9.5, encontra a célula mais parecida em E8.5
e calcula seu vetor de deslocamento (velocidade). Extrapola Δt à frente.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_cell_velocities(
    emb_e85: np.ndarray,
    emb_e95: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Para cada célula em E9.5, calcula a velocidade como:
        v_i = emb_e95_i - mean(kNN de emb_e95_i em emb_e85)

    Isso é análogo ao RNA Velocity, mas no espaço latente do LLM.
    k=5 para suavizar ruído de matching.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb_e85)
    _, indices = nbrs.kneighbors(emb_e95)  # (n_e95, k)

    # Centróide dos k vizinhos em E8.5 para cada célula de E9.5
    e85_anchors = emb_e85[indices].mean(axis=1)  # (n_e95, embed_dim)

    velocities = emb_e95 - e85_anchors  # vetor de "evolução" por célula
    return velocities  # (n_e95, embed_dim)


def extrapolate_to_e105(
    emb_e95: np.ndarray,
    velocities: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Aplica a velocidade para projetar E9.5 → E10.5.
    scale=1.0 assume intervalo temporal uniforme entre estágios.

    Normaliza após extrapolação para manter na mesma hiperesfera
    (importante se normalize_embeddings=True no encoder).
    """
    emb_e105 = emb_e95 + scale * velocities

    # Re-normalizar (L2) para manter consistência com o espaço treinado
    norms = np.linalg.norm(emb_e105, axis=1, keepdims=True)
    emb_e105 = emb_e105 / np.clip(norms, 1e-8, None)

    return emb_e105  # (n_e95, embed_dim)