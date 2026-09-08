"""
Decodifica de volta para o espaço de expressão as células de validação
que são de E9.5 (só essas, porque o `score` do veckit para a Task 1
espera um target/reference reais — ver docstring de run_veckit_score) e
roda a métrica oficial como checagem de sanidade da reconstrução.

Importante: isso NÃO é uma previsão real de E10.5/E12.5. É reconstrução
(decode(encode(x)) tentando voltar em x) sobre células de E9.5 que já
existem de verdade — serve só para validar se o embedding do scGPT
preserva informação suficiente pro decoder reconstruir a célula.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .decoder import ExpressionDecoder

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def decode_to_adata(
    model: ExpressionDecoder,
    embeddings: np.ndarray,
    reference_adata: ad.AnnData,
    obs_subset: Optional[pd.DataFrame] = None,
    device: str = "cpu",
) -> ad.AnnData:
    """Roda o decoder e empacota a saída como AnnData no mesmo eixo de
    genes (var) de `reference_adata`, pronto para o veckit.score."""
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32, device=device)
        pred = model(x).cpu().numpy()

    out = ad.AnnData(
        X=pred.astype(np.float32),
        var=reference_adata.var.copy(),
        obs=obs_subset.copy() if obs_subset is not None else None,
    )
    out.var_names = reference_adata.var_names
    return out


def run_veckit_score(
    prediction_path: PathLike,
    target_path: PathLike,
    reference_path: PathLike,
    task: str = "T1",
) -> dict:
    """Chama `veckit.score` exatamente como no README do projeto:
    input = sua predição, target = E9.5 real, reference = E8.5 real.
    """
    from veckit import score

    result = score(
        task=task,
        input=str(prediction_path),
        target=str(target_path),
        reference=str(reference_path),
    )
    logger.info("Métricas veckit: %s", result["metrics"])
    return result["metrics"]