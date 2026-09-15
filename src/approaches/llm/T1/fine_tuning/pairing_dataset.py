"""
Dataset de treino com pareamento probabilístico: em vez de sempre usar o
mesmo parceiro fixo (argmax) para cada célula de E8.5, sorteia um parceiro
entre os top-k mais prováveis do plano de transporte, respeitando os pesos.

Como o DataLoader chama __getitem__ de novo a cada época (com shuffle), o
parceiro sorteado muda de época para época, dando mais variedade de pares
pro modelo aprender.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class StochasticPairedDataset(Dataset):
    def __init__(self, x_input_e85, x_target_e95_full, topk_indices, topk_pesos):
        self.x_input_e85 = x_input_e85
        self.x_target_e95_full = x_target_e95_full
        self.topk_indices = topk_indices
        self.topk_pesos = topk_pesos

    def __len__(self):
        return self.x_input_e85.shape[0]

    def __getitem__(self, idx):
        parceiros_possiveis = self.topk_indices[idx]
        pesos = self.topk_pesos[idx]
        parceiro_sorteado = np.random.choice(parceiros_possiveis, p=pesos)

        x_in = torch.tensor(self.x_input_e85[idx], dtype=torch.float32)
        x_out = torch.tensor(self.x_target_e95_full[parceiro_sorteado], dtype=torch.float32)
        return x_in, x_out