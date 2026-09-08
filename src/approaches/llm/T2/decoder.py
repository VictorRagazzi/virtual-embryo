"""
Decoder: MLP simples que mapeia Y_512 (embedding de célula do scGPT) de
volta para o vetor de expressão gênica completo (~32k genes).

Por que MLP e não algo mais sofisticado: a pergunta que a ideia está
testando é "Y_512 carrega informação suficiente pra reconstruir a
célula?" — um decoder simples é o teste mais direto disso; se nem um MLP
razoavelmente largo reconstrói bem, um decoder mais complexo não vai
consertar uma perda de informação que já aconteceu no encoder.

Softplus na saída porque a expressão está em escala log1p (sempre >= 0);
sem isso o modelo poderia "prever" valores negativos sem sentido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn

PathLike = Union[str, Path]


class ExpressionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 32285,
        hidden_dims: Sequence[int] = (1024, 2048),
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

        # guardado para poder reconstruir a arquitetura ao carregar do disco
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

    def forward(self, y512: torch.Tensor) -> torch.Tensor:
        return self.net(y512)

    def save(self, path: PathLike, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: PathLike, map_location: str = "cpu") -> tuple["ExpressionDecoder", dict]:
        ckpt = torch.load(path, map_location=map_location)
        model = cls(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            hidden_dims=ckpt["hidden_dims"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt.get("extra", {})