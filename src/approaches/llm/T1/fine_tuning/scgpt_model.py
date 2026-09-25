"""
Modelo simplificado, compatível com os pesos do checkpoint scGPT (best_model.pt).

A arquitetura original do scGPT usa "flash attention", mas as chaves salvas no
checkpoint (self_attn.Wqkv, norm1, norm2, linear1, linear2) são equivalentes às
do nn.TransformerEncoderLayer padrão do PyTorch. Por isso, reconstruímos o
modelo só com blocos padrão do PyTorch e "traduzimos" os nomes dos pesos na
hora de carregar. Isso evita depender da biblioteca flash-attn.
"""
import torch
import torch.nn as nn

class ValueEncoder(nn.Module):
    """Transforma o valor (binned) de expressão de cada gene em um vetor de embedding."""

    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values):
        # values: (batch, n_genes) -> (batch, n_genes, 1)
        x = values.unsqueeze(-1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.norm(x)


class ExprDecoder(nn.Module):
    """Recebe o vetor de saída do transformer para cada gene e prevê sua expressão."""

    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class ScGPTRegressor(nn.Module):
    """
    Entrada: ids dos genes + valor (binned) de expressão de cada gene, para uma célula.
    Saída: expressão prevista de cada gene, na célula do estágio seguinte.
    """

    def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout):
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.enc_norm = nn.LayerNorm(d_model)
        self.flag_encoder = nn.Embedding(2, d_model)  # 0 = valor dado, 1 = valor mascarado (não usado aqui)
        self.value_encoder = ValueEncoder(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.decoder = ExprDecoder(d_model)

    def forward(self, gene_ids, values, pad_mask):
        return self.decoder(self.encode(gene_ids, values, pad_mask))

    def encode(self, gene_ids, values, pad_mask):
        """Retorna os estados contextualizados antes do decoder por gene."""
        gene_emb = self.enc_norm(self.gene_embedding(gene_ids))
        value_emb = self.value_encoder(values)
        flag_emb = self.flag_encoder(torch.zeros_like(gene_ids))

        x = gene_emb + value_emb + flag_emb
        return self.transformer_encoder(x, src_key_padding_mask=pad_mask)


def _remap_key(key):
    """Traduz os nomes de pesos do checkpoint original para os nomes usados aqui."""
    key = key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
    key = key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
    key = key.replace("encoder.embedding.weight", "gene_embedding.weight")
    key = key.replace("encoder.enc_norm", "enc_norm")
    return key


def load_pretrained_weights(model, checkpoint_path):
    """Carrega os pesos do best_model.pt no modelo, ignorando o que não bate."""
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in raw_state_dict:
        raw_state_dict = raw_state_dict["model"]

    renamed_state_dict = {_remap_key(k): v for k, v in raw_state_dict.items()}

    result = model.load_state_dict(renamed_state_dict, strict=False)
    print(f"[scGPT] pesos carregados. Faltando (ficam com init aleatória): {len(result.missing_keys)}")
    print(f"[scGPT] pesos do checkpoint não usados (ex: mvc_decoder): {len(result.unexpected_keys)}")
    return model