"""Encoder pré-treinado com regressão temporal direta ou residual."""

import torch
from torch import nn


class TemporalGeneformer(nn.Module):
    def __init__(self, encoder, output_genes, mode="delta", trainable_layers=2):
        super().__init__()
        if mode not in {"delta", "direct"}:
            raise ValueError("Modo deve ser delta ou direct.")
        layers = encoder.encoder.layer
        if not 0 <= trainable_layers <= len(layers):
            raise ValueError("Quantidade de camadas ajustáveis fora do intervalo.")
        self.encoder = encoder
        self.mode = mode
        self.trainable_layers = trainable_layers
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        if trainable_layers:
            for layer in layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        self.head = nn.Sequential(nn.Linear(encoder.config.hidden_size + 2, 128),
                                  nn.GELU(), nn.Linear(128, output_genes))

    def train(self, mode=True):
        super().train(mode)
        # Desativa dropout na parte congelada para tornar sua representação estável.
        self.encoder.embeddings.eval()
        frozen_count = len(self.encoder.encoder.layer) - self.trainable_layers
        for layer in self.encoder.encoder.layer[:frozen_count]:
            layer.eval()
        return self

    def forward(self, input_ids, mask, expression, times):
        hidden = self.encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        change = self.head(torch.cat([pooled, times / 10.0], dim=1))
        return expression + change if self.mode == "delta" else change


def load_pretrained_encoder(path):
    from transformers import BertModel

    encoder, loading = BertModel.from_pretrained(
        path, add_pooling_layer=False, output_loading_info=True, local_files_only=True,
    )
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise ValueError(f"Checkpoint incompleto/incompatível: {loading}")
    return encoder
