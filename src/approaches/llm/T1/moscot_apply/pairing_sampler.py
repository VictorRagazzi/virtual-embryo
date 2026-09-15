"""Conversão da matriz de transporte (OT) em uma tabela de pares (E8.5 -> E9.5).

A matriz de transporte T (n_source x n_target) do OT não-balanceado NÃO é uma
matriz de permutação: cada linha i (célula de E8.5) tem uma distribuição de massa
sobre as células de E9.5 (colunas), e a soma da linha pode ser < 1 (célula sem
"descendente" claro no next timepoint, sob a ótica do OT não-balanceado).

Aqui tratamos T[i, :] / T[i, :].sum() como uma distribuição de probabilidade
P(alvo = j | fonte = i), e oferecemos três formas de extrair pares dela.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PairingMode = Literal["sample", "argmax", "top_k"]


def transport_to_pairs(
    T: np.ndarray,
    source_names: pd.Index,
    target_names: pd.Index,
    mode: PairingMode = "sample",
    n_samples: int = 1,
    min_row_mass: float = 1e-8,
    random_state: int = 0,
) -> pd.DataFrame:
    """Converte a matriz de transporte em uma tabela de pares (source, target, weight).

    Parâmetros
    ----------
    T : matriz de transporte densa, shape (n_source, n_target).
    source_names, target_names : obs_names (na mesma ordem usada para montar T).
    mode :
        "sample"  - para cada célula fonte, sorteia `n_samples` alvos com
                    probabilidade proporcional à linha de T (pareamento
                    probabilístico, amostragem com reposição). Rode com
                    n_samples=1 para "uma célula associada por célula fonte",
                    ou n_samples>1 para gerar múltiplos pares (aumento de dados).
        "argmax"  - escolhe deterministicamente o alvo de maior peso por célula
                    fonte (equivalente ao "melhor par").
        "top_k"   - mantém os `n_samples` alvos de maior peso por célula fonte,
                    com pesos renormalizados para somarem 1 dentro do grupo
                    (sem aleatoriedade).
    min_row_mass : células fonte cuja soma de linha em T for menor que este
        valor são descartadas do pareamento (massa de transporte desprezível -
        nenhum alvo plausível foi encontrado pelo OT).
    random_state : semente para reprodutibilidade do modo "sample".

    Retorna
    -------
    DataFrame com colunas:
        source_idx, target_idx  - índices posicionais em T
        source_cell, target_cell - obs_names correspondentes
        weight                   - peso do par (probabilidade normalizada na linha)
        row_mass                 - massa total da linha da célula fonte em T
    """
    rng = np.random.default_rng(random_state)
    T = np.asarray(T)
    row_mass = T.sum(axis=1)

    valid_rows = np.where(row_mass > min_row_mass)[0]
    dropped = T.shape[0] - len(valid_rows)
    if dropped:
        logger.warning(
            "%d/%d células de E8.5 com massa de transporte desprezível "
            "(< %.2e) foram descartadas do pareamento.",
            dropped, T.shape[0], min_row_mass,
        )

    records = []
    for i in valid_rows:
        row = T[i]
        probs = row / row.sum()

        if mode == "argmax":
            j_arr = np.array([int(np.argmax(row))])
            w_arr = np.array([1.0])
        elif mode == "sample":
            j_arr = rng.choice(len(row), size=n_samples, replace=True, p=probs)
            w_arr = probs[j_arr]
        elif mode == "top_k":
            j_arr = np.argsort(row)[::-1][:n_samples]
            w = row[j_arr]
            w_arr = w / w.sum()
        else:
            raise ValueError(f"mode desconhecido: {mode!r} (use 'sample', 'argmax' ou 'top_k')")

        for j, w in zip(j_arr, w_arr):
            records.append(
                (int(i), int(j), source_names[i], target_names[j], float(w), float(row_mass[i]))
            )

    df = pd.DataFrame(
        records,
        columns=["source_idx", "target_idx", "source_cell", "target_cell", "weight", "row_mass"],
    )
    logger.info(
        "Pareamento gerado: %d pares a partir de %d células fonte válidas (mode=%s, n_samples=%d).",
        len(df), len(valid_rows), mode, n_samples,
    )
    return df