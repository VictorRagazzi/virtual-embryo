"""
Aplica o modelo já fine-tunado sobre as células de E9.5 para gerar a
predição de E10.5.

Rodar com:
    uv run python -m src.approaches.llm.T1.fine_tuning.predict_e10
"""
import anndata as ad
import numpy as np
import scanpy as sc
import torch

from .config import Config
config = Config()

from .data_prep import bin_expression_matrix, fill_missing_genes
from .scgpt_model import ScGPTRegressor
from .vocab_utils import gene_to_vocab_id, load_vocab

# tamanho do lote usado só na predição (não precisa ser igual ao do treino).
# se ainda faltar memória, diminua esse número (ex: 8 ou 4).
PREDICT_BATCH_SIZE = 128


def predict_in_batches(model, gene_ids, x_input, batch_size):
    n_celulas = x_input.shape[0]
    saidas = []

    with torch.no_grad():
        for inicio in range(0, n_celulas, batch_size):
            fim = min(inicio + batch_size, n_celulas)
            lote_valores = x_input[inicio:fim]
            lote_gene_ids = gene_ids.unsqueeze(0).expand(lote_valores.shape[0], -1)
            lote_pad_mask = torch.zeros_like(lote_gene_ids, dtype=torch.bool)

            saida = model(lote_gene_ids, lote_valores, lote_pad_mask)
            saida = torch.clamp(saida, min=0.0)   # <-- expressão não pode ser negativa
            saidas.append(saida.numpy())

            print(f"[predição] células {fim}/{n_celulas} processadas")

    return np.concatenate(saidas, axis=0)


def main():
    checkpoint = torch.load(config.FINETUNED_PATH, map_location="cpu")
    gene_list = checkpoint["gene_list"]
    vocab = load_vocab(config.VOCAB_PATH)
    gene_ids = torch.tensor([gene_to_vocab_id(g, vocab) for g in gene_list], dtype=torch.long)

    model = ScGPTRegressor(
        vocab_size=len(vocab),
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        d_hid=config.D_HID,
        nlayers=config.N_LAYERS,
        dropout=config.DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    adata95 = sc.read_h5ad(config.E95_PATH_SUB2000)
    adata95_sub = adata95[:, gene_list].copy()

    x_input = bin_expression_matrix(adata95_sub.X, config.N_BINS)
    x_input = torch.tensor(x_input)

    predicao = predict_in_batches(model, gene_ids, x_input, PREDICT_BATCH_SIZE)

    # completa os genes que o modelo não previu (fora do gene_list) com o
    # valor de E9.5, usando o E9.5 completo (todos os genes) como referência
    x_final = fill_missing_genes(adata95, gene_list, predicao)

    adata_pred = ad.AnnData(
        X=x_final,
        obs=adata95.obs.copy(),
        var=adata95.var.copy(),
    )
    adata_pred.obs_names = [f"pred_e10_5_{i}" for i in range(adata_pred.n_obs)]

    adata_pred.write_h5ad(config.PRED_PATH)
    print(f"[predição] E10.5 prevista salva em {config.PRED_PATH}")


if __name__ == "__main__":
    main()