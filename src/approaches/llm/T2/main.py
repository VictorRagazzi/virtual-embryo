"""
CLI do experimento: encoder scGPT (congelado) -> Y_512 -> decoder (MLP)
treinado para reconstruir a expressão original, com split estratificado
por tipo celular e checagem de sanidade via veckit sobre a validação
E9.5-only.

Exemplos (rodar da raiz do projeto virtualembryo/, via uv):

    # roda tudo: embed -> split -> treino -> score
    uv run python -m src.scgpt_decoder.main all \\
        --e85 data/E85.h5ad --e95 data/E95.h5ad \\
        --scgpt-model-dir models/scgpt_human \\
        --celltype-key celltype

    # só recalcula/cacheia Y_512 (útil pra rodar 1x numa GPU maior e
    # depois treinar o decoder várias vezes sem repetir o encoder)
    uv run python -m src.scgpt_decoder.main embed \\
        --e85 data/E85.h5ad --e95 data/E95.h5ad --scgpt-model-dir models/scgpt_human

    # só treina o decoder (usa o cache de Y_512, se existir)
    uv run python -m src.scgpt_decoder.main train --epochs 200

    # só avalia um decoder já treinado em models/
    uv run python -m src.scgpt_decoder.main evaluate

O checkpoint pré-treinado do scGPT precisa ser baixado manualmente (não
vem via pip) e apontado em --scgpt-model-dir. Veja o comentário no topo
de scgpt_embed.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np

from .dataset import EmbeddingToExpressionDataset, stratified_split_indices
from .decoder import ExpressionDecoder
from .evaluate import decode_to_adata, run_veckit_score
from .scgpt_embed import get_or_compute_embeddings
from .train import train_decoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_concatenated(e85_path: str, e95_path: str, celltype_key: str) -> ad.AnnData:
    a85 = ad.read_h5ad(e85_path)
    a95 = ad.read_h5ad(e95_path)
    a85.obs["stage"] = "E8.5"
    a95.obs["stage"] = "E9.5"
    joint = ad.concat([a85, a95], join="inner", index_unique="-")
    if celltype_key not in joint.obs.columns:
        raise KeyError(
            f"'{celltype_key}' não está em adata.obs (colunas disponíveis: "
            f"{list(joint.obs.columns)}). Passe --celltype-key correto."
        )
    return joint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["embed", "train", "evaluate", "all"])

    # dados
    p.add_argument("--e85", default="data/E85.h5ad")
    p.add_argument("--e95", default="data/E95.h5ad")
    p.add_argument("--celltype-key", default="celltype", help="coluna em adata.obs com o tipo celular")

    # encoder (scGPT)
    p.add_argument("--scgpt-model-dir", default="models/scgpt_human")
    p.add_argument("--gene-col", default="index")
    p.add_argument("--max-length", type=int, default=1200)
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")

    # split
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)

    # decoder / treino
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)

    # saída
    p.add_argument("--models-dir", default="models")
    p.add_argument("--decoder-name", default="scgpt_expression_decoder.pt")
    p.add_argument("--cache-dir", default="data/scgpt_cache")
    p.add_argument("--predictions-out", default="data/prediction_e9_5_decoder_val.h5ad")

    return p


def main() -> None:
    args = build_parser().parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    decoder_path = models_dir / args.decoder_name

    joint = load_concatenated(args.e85, args.e95, args.celltype_key)
    cell_types = joint.obs[args.celltype_key].to_numpy()
    stage = joint.obs["stage"].to_numpy()

    # --- 1) encoder: Y_512 (com cache em disco) ---
    y512 = None
    if args.stage in ("embed", "train", "all"):
        y512 = get_or_compute_embeddings(
            joint,
            cache_path=Path(args.cache_dir) / "joint_y512.npy",
            model_dir=args.scgpt_model_dir,
            gene_col=args.gene_col,
            max_length=args.max_length,
            batch_size=args.embed_batch_size,
            device=args.device,
        )
        logger.info("Y_512: %s", y512.shape)

    if args.stage == "embed":
        return

    # a partir daqui (train/evaluate/all) precisamos de Y_512 mesmo que
    # tenha vindo do cache em vez de ter sido calculado agora
    if y512 is None:
        y512 = get_or_compute_embeddings(
            joint,
            cache_path=Path(args.cache_dir) / "joint_y512.npy",
            model_dir=args.scgpt_model_dir,
            gene_col=args.gene_col,
            max_length=args.max_length,
            batch_size=args.embed_batch_size,
            device=args.device,
        )

    expression = np.asarray(
        joint.X.toarray() if hasattr(joint.X, "toarray") else joint.X, dtype=np.float32
    )

    # --- 2) split estratificado por tipo celular ---
    train_idx, val_idx = stratified_split_indices(cell_types, args.val_fraction, args.split_seed)
    logger.info(
        "split estratificado por '%s': %d treino / %d validação",
        args.celltype_key, len(train_idx), len(val_idx),
    )

    train_ds = EmbeddingToExpressionDataset(y512[train_idx], expression[train_idx])
    val_ds = EmbeddingToExpressionDataset(y512[val_idx], expression[val_idx])

    # --- 3) treino do decoder ---
    if args.stage in ("train", "all"):
        _model, _history = train_decoder(
            train_ds,
            val_ds,
            input_dim=y512.shape[1],
            output_dim=expression.shape[1],
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            save_path=decoder_path,
        )

    if args.stage == "train":
        return

    # --- 4) avaliação: só a fatia E9.5 da validação, via veckit ---
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Nenhum decoder treinado em {decoder_path}; rode 'train' ou 'all' primeiro."
        )
    model, _extra = ExpressionDecoder.load(decoder_path)

    val_stage = stage[val_idx]
    e95_mask = val_stage == "E9.5"
    if e95_mask.sum() == 0:
        raise RuntimeError(
            "Nenhuma célula E9.5 caiu na fatia de validação; tente outro --split-seed."
        )
    e95_val_idx = val_idx[e95_mask]

    pred_adata = decode_to_adata(
        model,
        y512[e95_val_idx],
        reference_adata=joint,
        obs_subset=joint.obs.iloc[e95_val_idx],
    )

    out_path = Path(args.predictions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    logger.info("Predições (reconstrução) salvas em %s (%d células)", out_path, pred_adata.n_obs)

    metrics = run_veckit_score(out_path, target_path=args.e95, reference_path=args.e85)
    print("\nMétricas veckit (checagem de sanidade da reconstrução, validação E9.5-only):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()