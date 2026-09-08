"""Loop de treino do ExpressionDecoder. Reutilizável: chame
`train_decoder(...)` do main.py, de um notebook, ou de outro script."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import EmbeddingToExpressionDataset
from .decoder import ExpressionDecoder

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def train_decoder(
    train_ds: EmbeddingToExpressionDataset,
    val_ds: EmbeddingToExpressionDataset,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (1024, 2048),
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cuda",
    save_path: PathLike | None = None,
) -> tuple[ExpressionDecoder, dict]:
    """Treina com MSE (reconstrução em escala log1p) + early stopping na
    val loss. Guarda os melhores pesos (menor val_loss), não os últimos.
    """
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA indisponível — treinando o decoder na CPU (vai ser lento).")

    model = ExpressionDecoder(input_dim, output_dim, hidden_dims, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(
            "epoch %3d/%d  train_mse=%.4f  val_mse=%.4f", epoch, epochs, train_loss, val_loss
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                logger.info(
                    "Early stopping na epoch %d (sem melhora por %d epochs)", epoch, patience
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")

    if save_path is not None:
        model.save(save_path, extra={"best_val_mse": best_val, "history": history})
        logger.info("Decoder salvo em %s", save_path)

    return model, history