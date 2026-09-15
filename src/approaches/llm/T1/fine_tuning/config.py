"""
Configuração central do pipeline de fine-tuning do scGPT.
Todos os caminhos e hiperparâmetros ficam aqui para facilitar ajustes.
"""
from pathlib import Path

# --- pastas do projeto ---
class Config:
    DATA_DIR = Path("data")
    MODEL_DIR = Path("models/scGPT_heart")
    PREDICTIONS = Path("predictions")
    OUT_DIR = Path("src/approaches/llm/T1/fine_tuning/outputs")

    # --- arquivos de entrada ---
    E85_PATH         = DATA_DIR / "E85.h5ad"
    E95_PATH         = DATA_DIR / "E95.h5ad"
    E85_PATH_SUB2000 = DATA_DIR / "E85_sub2000.h5ad"
    E95_PATH_SUB2000 = DATA_DIR / "E95_sub2000.h5ad"
    PAIRED_PATH = DATA_DIR / "e85_e95_ot_pairs.h5ad"
    TRANSPORT_PATH = DATA_DIR / "e85_e95_ot_pairs_transport.npz"


    DE_GENES_PATH = DATA_DIR / "de_genes_e85_e95.json"   # saída do select_de_genes.py
    TOPK_PARTNERS = 5                                     # quantos parceiros candidatos por célula de E8.5
    CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"
    VOCAB_PATH = MODEL_DIR / "vocab.json"

    # --- arquivos de saída ---
    FINETUNED_PATH = MODEL_DIR / "scgpt_finetuned.pt"
    PRED_PATH = PREDICTIONS / "prediction_e10_5.h5ad"

    # --- hiperparâmetros do modelo (copiados do args.json original do scGPT) ---
    N_GENES = 1200       # limite de genes por célula (max_seq_len do modelo)
    N_BINS = 51          # número de bins de expressão usados na entrada
    D_MODEL = 512
    N_HEAD = 8
    N_LAYERS = 12
    D_HID = 512
    DROPOUT = 0.2

    # --- hiperparâmetros do treino ---
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
