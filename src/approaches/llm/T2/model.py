"""Encoder pré-treinado com regressão temporal direta ou residual."""

import torch
from torch import nn


class TemporalGeneformer(nn.Module):
    def __init__(self, encoder, output_genes, trainable_layers=2, expression_input="none", mode="delta"):
        super().__init__()
        if mode not in {"delta", "direct", "velocity"}:
            raise ValueError("Modo deve ser delta ou direct.")
        if expression_input not in {"none", "projected"}:
            raise ValueError("expression_input deve ser none ou projected.")
        layers = encoder.encoder.layer
        if not 0 <= trainable_layers <= len(layers):
            raise ValueError("Quantidade de camadas ajustáveis fora do intervalo.")
        self.encoder = encoder
        self.mode = mode
        self.expression_input = expression_input
        self.trainable_layers = trainable_layers
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        if trainable_layers:
            for layer in layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        # A projeção de 16 dimensões fornece magnitudes por gene sem duplicar
        # a grande matriz de saída da cabeça.
        self.expression_projection = (nn.Linear(output_genes, 16, bias=False)
                                      if expression_input == "projected" else None)
        head_input = encoder.config.hidden_size + 2 + (16 if expression_input == "projected" else 0)
        self.head = nn.Sequential(nn.Linear(head_input, 128),
                                  nn.GELU(), nn.Linear(128, output_genes))

    def train(self, mode=True):
        super().train(mode)
        # Desativa dropout na parte congelada para tornar sua representação estável.
        self.encoder.embeddings.eval()
        frozen_count = len(self.encoder.encoder.layer) - self.trainable_layers
        for layer in self.encoder.encoder.layer[:frozen_count]:
            layer.eval()
        return self

    def forward(self, input_ids, mask, expression, times, alpha=1.0):
        hidden = self.encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        features = [pooled, times / 10.0]
        if self.expression_projection is not None:
            features.append(self.expression_projection(expression))
        result = self.head(torch.cat(features, dim=1))
        if self.mode == "velocity":  # Checkpoints da arquitetura anterior.
            return expression + (alpha * times[:, 1:2]) * result
        if self.mode == "delta":
            return expression + alpha * result
        return expression + alpha * (result - expression)


def load_pretrained_encoder(path):
    from transformers import BertModel

    encoder, loading = BertModel.from_pretrained(
        path, add_pooling_layer=False, output_loading_info=True, local_files_only=True,
    )
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise ValueError(f"Checkpoint incompleto/incompatível: {loading}")
    return encoder
