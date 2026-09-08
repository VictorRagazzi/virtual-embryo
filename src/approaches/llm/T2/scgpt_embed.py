"""
Encoder: roda o scGPT pré-treinado (congelado, sem fine-tuning) sobre as
células e retorna Y_512 — o embedding de célula do scGPT (512 dimensões
no checkpoint whole-human, independente do número de genes de entrada).

Requisitos (instalar no servidor, não incluso aqui por serem pesados):
    uv add scgpt torch scanpy anndata

Checkpoint pré-treinado: o scGPT não distribui os pesos via pip. Baixe o
checkpoint "whole-human" (ou o equivalente que você tiver disponível) do
repositório oficial (bowang-lab/scGPT) e aponte `model_dir` para a pasta
que contém `args.json`, `vocab.json` e `best_model.pt`.

Aviso sobre espécie: os checkpoints públicos do scGPT são treinados em
vocabulário HUMANO. Nossos dados são de camundongo. Como a maioria dos
ortólogos 1:1 mouse<->human tem o mesmo símbolo (só muda a caixa:
"Nppa" vs "NPPA"), fazemos um upper-case ingênuo dos gene symbols antes
de embedar (`humanize_mouse_symbols`). Isso é uma aproximação: genes sem
ortólogo direto ou com símbolo diferente simplesmente não batem no
vocabulário e são descartados automaticamente pelo `embed_data` (ele já
loga quantos genes bateram). Para essa validação de ideia é aceitável;
se a ideia for adiante, vale trocar por um mapeamento de ortólogos via
biomart/MGI em vez do upper-case.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

if platform.system() == "Windows" and not hasattr(os, "sched_getaffinity"):
    # scgpt.tasks.cell_emb.get_batch_cell_embeddings chama
    # os.sched_getaffinity(0) pra decidir o num_workers do DataLoader — é
    # uma API exclusiva do Linux/POSIX e não existe no Windows, o que
    # derruba o embed_data com AttributeError antes mesmo de rodar o
    # modelo. Repõe uma implementação equivalente: "todas as CPUs
    # disponíveis para este processo", usando os.cpu_count() como proxy.
    def _sched_getaffinity_shim(pid: int) -> set[int]:  # noqa: ARG001
        return set(range(os.cpu_count() or 1))

    os.sched_getaffinity = _sched_getaffinity_shim  # type: ignore[attr-defined]
    logger.info("Windows detectado: aplicado shim de os.sched_getaffinity para o scGPT.")

PathLike = Union[str, Path]


def humanize_mouse_symbols(var_names: Sequence[str]) -> list[str]:
    """Upper-case ingênuo para aproximar símbolos de gene de camundongo
    dos símbolos humanos usados no vocabulário do scGPT."""
    return [str(g).upper() for g in var_names]


def compute_cell_embeddings(
    adata: ad.AnnData,
    model_dir: PathLike,
    gene_col: str = "index",
    max_length: int = 1200,
    batch_size: int = 64,
    device: str = "cuda",
    obs_to_save: Optional[list[str]] = None,
    humanize_genes: bool = True,
) -> ad.AnnData:
    """Passa `adata` pelo scGPT e devolve um AnnData cujo .X é Y_512
    (n_células, 512), na mesma ordem de linhas de `adata`.

    Não faz fine-tuning: usa o modelo pré-treinado como está (feature
    extractor / encoder), que é exatamente o papel de "encoder" pedido.
    """
    import scgpt as scg  # import local: dependência pesada e opcional

    work = adata.copy()
    if humanize_genes:
        work.var_names = humanize_mouse_symbols(work.var_names)
        gene_col = "index"

    embedded = scg.tasks.embed_data(
        work,
        model_dir=str(model_dir),
        gene_col=gene_col,
        max_length=max_length,
        batch_size=batch_size,
        obs_to_save=obs_to_save,
        device=device,
        return_new_adata=True,
    )
    logger.info("Y_512 calculado: %s células x %s dimensões", *embedded.shape)
    return embedded


def get_or_compute_embeddings(
    adata: ad.AnnData,
    cache_path: PathLike,
    model_dir: PathLike,
    **kwargs,
) -> np.ndarray:
    """Cacheia Y_512 em disco (.npy) para não recalcular o encoder (a
    etapa mais cara) toda vez que você reroda o treino do decoder."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Lendo embeddings cacheados de %s", cache_path)
        return np.load(cache_path)

    embedded = compute_cell_embeddings(adata, model_dir, **kwargs)
    y512 = np.asarray(embedded.X, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, y512)
    logger.info("Y_512 salvo em %s", cache_path)
    return y512