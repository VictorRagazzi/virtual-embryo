"""
Split estratificado por tipo celular (80/20 dentro de cada tipo, como no
exemplo: sangue/muscular/neural cada um perdendo ~20% pra validação) e o
Dataset que empareia Y_512 (entrada do decoder) com a expressão original
(saída/target do decoder).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def stratified_split_indices(
    cell_types: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (train_idx, val_idx) tal que cada tipo celular contribui
    ~val_fraction dos seus próprios exemplos para a validação — é o
    "16.000/24.000/40.000 vão pra treino" do seu exemplo, generalizado
    para N tipos celulares e qualquer val_fraction.

    Tipos celulares com menos de 2 células não podem ser estratificados
    (train_test_split exige >=2 por classe) e vão inteiros para o treino.
    """
    cell_types = np.asarray(cell_types)
    idx_all = np.arange(len(cell_types))

    counts = {ct: int((cell_types == ct).sum()) for ct in np.unique(cell_types)}
    singleton_types = [ct for ct, n in counts.items() if n < 2]
    if singleton_types:
        import logging

        logging.getLogger(__name__).warning(
            "Tipos celulares com <2 células (indo inteiros para treino): %s",
            singleton_types,
        )

    keep_mask = ~np.isin(cell_types, singleton_types)
    idx_splittable = idx_all[keep_mask]
    idx_forced_train = idx_all[~keep_mask]

    train_idx, val_idx = train_test_split(
        idx_splittable,
        test_size=val_fraction,
        stratify=cell_types[idx_splittable],
        random_state=seed,
    )
    train_idx = np.concatenate([train_idx, idx_forced_train])
    return np.sort(train_idx), np.sort(val_idx)


class EmbeddingToExpressionDataset(Dataset):
    """Par (Y_512, expressão original) para treinar o decoder."""

    def __init__(self, embeddings: np.ndarray, expression: np.ndarray):
        assert embeddings.shape[0] == expression.shape[0], (
            f"embeddings tem {embeddings.shape[0]} linhas mas expression tem "
            f"{expression.shape[0]}"
        )
        self.x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(expression), dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]