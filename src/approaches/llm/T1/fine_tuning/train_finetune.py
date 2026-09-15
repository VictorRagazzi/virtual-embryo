"""
Fine-tuning do scGPT: aprende a prever a expressão de uma célula em t+1
a partir da expressão da célula pareada em t (estilo perturb-GEP do scGPT).

Mudanças em relação à primeira versão:
  - genes selecionados por diferença de expressão E8.5->E9.5 (select_de_genes),
    em vez de variância bruta;
  - pareamento probabilístico: a cada época, sorteia o parceiro de E9.5 entre
    os top-k mais prováveis do plano de transporte, em vez de sempre usar o
    mesmo parceiro fixo (argmax).

Rodar com:
    uv run python -m src.approaches.llm.T1.fine_tuning.train_finetune
"""
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from .config import Config
config = Config()

from .data_prep import bin_expression_matrix, build_topk_partners, to_dense
from .pairing_dataset import StochasticPairedDataset
from .scgpt_model import ScGPTRegressor, load_pretrained_weights
from .vocab_utils import gene_to_vocab_id, load_vocab, select_de_genes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[treino] usando device: {device}")

    vocab = load_vocab(config.VOCAB_PATH)

    adata85 = sc.read_h5ad(config.E85_PATH)
    adata95 = sc.read_h5ad(config.E95_PATH)

    gene_list = select_de_genes(adata85, vocab, config.N_GENES, config.DE_GENES_PATH)
    gene_ids = torch.tensor([gene_to_vocab_id(g, vocab) for g in gene_list], dtype=torch.long)

    adata85_sub = adata85[:, gene_list]
    adata95_sub = adata95[:, gene_list]

    x_input_e85 = bin_expression_matrix(adata85_sub.X, config.N_BINS)
    x_target_e95_full = to_dense(adata95_sub.X)

    topk_indices, topk_pesos = build_topk_partners(config.TRANSPORT_PATH, k=config.TOPK_PARTNERS)

    assert topk_indices.shape[0] == x_input_e85.shape[0], (
        "Número de linhas do plano de transporte não bate com o número de "
        "células de E8.5. Confira se E85.h5ad é o mesmo arquivo usado para "
        "calcular o plano de transporte."
    )

    dataset = StochasticPairedDataset(x_input_e85, x_target_e95_full, topk_indices, topk_pesos)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = ScGPTRegressor(
        vocab_size=len(vocab),
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        d_hid=config.D_HID,
        nlayers=config.N_LAYERS,
        dropout=config.DROPOUT,
    )
    model = load_pretrained_weights(model, config.CHECKPOINT_PATH)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(config.EPOCHS):
        perda_total = 0.0
        for valores_b, alvo_b in loader:
            valores_b = valores_b.to(device)
            alvo_b = alvo_b.to(device)
            gene_ids_b = gene_ids.unsqueeze(0).expand(valores_b.shape[0], -1).to(device)
            pad_mask = torch.zeros_like(gene_ids_b, dtype=torch.bool)

            optimizer.zero_grad()
            predicao = model(gene_ids_b, valores_b, pad_mask)
            perda = loss_fn(predicao, alvo_b)
            perda.backward()
            optimizer.step()
            perda_total += perda.item()

        print(f"[treino] epoch {epoch + 1}/{config.EPOCHS} - perda média: {perda_total / len(loader):.4f}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "gene_list": gene_list},
        config.FINETUNED_PATH,
    )
    print(f"[treino] modelo salvo em {config.FINETUNED_PATH}")


if __name__ == "__main__":
    main()