"""Round-trip exploratório: scGPT congelado -> decoder linear -> expressão."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.approaches.llm.T1.fine_tuning.config import Config
from src.approaches.llm.T1.fine_tuning.data_prep import bin_expression_matrix
from src.approaches.llm.T1.fine_tuning.scgpt_model import ScGPTRegressor, load_pretrained_weights


class NonnegativeLinearDecoder(nn.Module):
    """Decoder configurável, auditável e com saída estritamente não negativa."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim:
            self.linear = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        else:
            self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(latent))


def set_seed(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sample_stage(path: Path, cells: int, seed: int) -> ad.AnnData:
    """Amostra células por leitura backed, sem carregar o estágio inteiro na RAM."""
    backed = ad.read_h5ad(path, backed="r")
    try:
        if not 1 <= cells <= backed.n_obs:
            raise ValueError(f"cells-per-stage deve estar em [1, {backed.n_obs}].")
        index = np.sort(np.random.default_rng(seed).choice(backed.n_obs, cells, replace=False))
        return backed[index].to_memory()
    finally:
        backed.file.close()


def select_input_genes(matrix, gene_names: list[str], vocab: dict[str, int], count: int) -> list[str]:
    """Seleciona genes de maior variância no treino que existem no vocabulário."""
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    variance = np.maximum(second_moment - mean**2, 0)
    candidates = [(float(variance[index]), gene) for index, gene in enumerate(gene_names) if gene in vocab or gene.upper() in vocab]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [gene for _, gene in candidates[:count]]
    if len(selected) != count:
        raise ValueError(f"Só {len(selected)} genes do painel existem no vocabulário; necessário: {count}.")
    return selected


def token_id(gene: str, vocab: dict[str, int]) -> int:
    """Resolve capitalização sem fazer inferência de ortologia entre espécies."""
    if gene in vocab:
        return vocab[gene]
    return vocab[gene.upper()]


def build_sequences(matrix, gene_names: list[str], selected_genes: list[str], vocab: dict[str, int], n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta [<cls>, genes] e valores binned para o encoder congelado."""
    positions = {gene: index for index, gene in enumerate(gene_names)}
    subset = matrix[:, [positions[gene] for gene in selected_genes]]
    binned = bin_expression_matrix(subset, n_bins)
    n_cells = binned.shape[0]
    ids = np.empty((n_cells, len(selected_genes) + 1), dtype=np.int64)
    values = np.empty((n_cells, len(selected_genes) + 1), dtype=np.float32)
    ids[:, 0] = vocab["<cls>"]
    ids[:, 1:] = np.asarray([token_id(gene, vocab) for gene in selected_genes], dtype=np.int64)
    values[:, 0] = -2.0  # pad_value do checkpoint para o token <cls>
    values[:, 1:] = binned
    return ids, values, np.zeros(ids.shape, dtype=bool)


def extract_embeddings(model: ScGPTRegressor, ids: np.ndarray, values: np.ndarray, mask: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            states = model.encode(torch.from_numpy(ids[start:stop]), torch.from_numpy(values[start:stop]), torch.from_numpy(mask[start:stop]))
            chunks.append(states[:, 0, :].cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def split_by_stage(stages: np.ndarray, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train, validation = [], []
    rng = np.random.default_rng(seed)
    for stage in np.unique(stages):
        positions = np.flatnonzero(stages == stage)
        shuffled = rng.permutation(positions)
        n_validation = max(1, int(round(len(positions) * validation_fraction)))
        validation.extend(shuffled[:n_validation])
        train.extend(shuffled[n_validation:])
    return np.sort(train), np.sort(validation)


def select_high_variance_gene_indices(matrix, count: int) -> np.ndarray:
    """Retorna índices de genes variáveis sem densificar a matriz completa."""
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    variance = np.maximum(second_moment - mean**2, 0)
    if not 2 <= count <= matrix.shape[1]:
        raise ValueError(f"covariance-genes deve estar em [2, {matrix.shape[1]}].")
    return np.argsort(-variance, kind="stable")[:count].astype(np.int64)


def normalized_covariance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compara covariâncias de genes, normalizada pela energia do alvo no batch.

    A diagonal preserva pressão sobre a variância; os termos fora da diagonal
    preservam covariação. A normalização mantém a escala comparável entre batches.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction e target devem ter o mesmo shape.")
    if prediction.shape[0] < 2:
        raise ValueError("São necessárias ao menos duas células para covariância.")
    divisor = prediction.shape[0] - 1
    centered_prediction = prediction - prediction.mean(dim=0, keepdim=True)
    centered_target = target - target.mean(dim=0, keepdim=True)
    covariance_prediction = centered_prediction.T @ centered_prediction / divisor
    covariance_target = centered_target.T @ centered_target / divisor
    target_energy = covariance_target.square().mean().detach().clamp_min(torch.finfo(target.dtype).eps)
    return F.mse_loss(covariance_prediction, covariance_target) / target_energy


def snap_to_reference_expression(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projeta cada saída na célula de referência mais próxima por distância L2.

    Este pós-processamento é intencionalmente uma recuperação por memória. Quando
    usado com referência que contém validação, é diagnóstico com vazamento, nunca
    uma métrica de generalização ou saída submetível.
    """
    if prediction.ndim != 2 or reference.ndim != 2 or prediction.shape[1] != reference.shape[1]:
        raise ValueError("prediction e reference devem ser matrizes 2D com o mesmo número de genes.")
    if not len(reference):
        raise ValueError("reference deve conter ao menos uma célula.")
    prediction_norm = np.einsum("ij,ij->i", prediction, prediction)[:, None]
    reference_norm = np.einsum("ij,ij->i", reference, reference)[None, :]
    squared_distances = prediction_norm + reference_norm - 2 * prediction @ reference.T
    nearest = np.argmin(squared_distances, axis=1)
    return reference[nearest].astype(np.float32, copy=True), nearest.astype(np.int64)


def add_oracle_residual(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Soma o resíduo conhecido da validação; somente teto diagnóstico com vazamento."""
    if prediction.shape != target.shape:
        raise ValueError("prediction e target devem ter o mesmo shape.")
    return (prediction + (target - prediction)).astype(np.float32, copy=False)


def population_metrics(prediction: np.ndarray, target: np.ndarray, train_target: np.ndarray) -> dict[str, float]:
    pseudobulk_pred, pseudobulk_true = prediction.mean(axis=0), target.mean(axis=0)
    baseline = np.broadcast_to(train_target.mean(axis=0), target.shape)
    top = np.argsort(train_target.var(axis=0))[-min(128, target.shape[1]):]
    covariance_pred = np.cov(prediction[:, top], rowvar=False)
    covariance_true = np.cov(target[:, top], rowvar=False)
    upper = np.triu_indices(len(top), k=1)
    covariance_correlation = np.corrcoef(covariance_pred[upper], covariance_true[upper])[0, 1]
    return {
        "decoder_mse": float(np.mean((prediction - target) ** 2)),
        "mean_profile_mse": float(np.mean((baseline - target) ** 2)),
        "pseudobulk_pearson": float(np.corrcoef(pseudobulk_pred, pseudobulk_true)[0, 1]),
        "variance_ratio": float(prediction.var(axis=0).mean() / (target.var(axis=0).mean() + 1e-12)),
        "gene_variance_spearman": float(spearmanr(prediction.var(axis=0), target.var(axis=0)).statistic),
        "sampled_covariance_pearson": float(covariance_correlation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e85", type=Path, default=Path("data/E85.h5ad"))
    parser.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    parser.add_argument("--checkpoint", type=Path, default=Config.CHECKPOINT_PATH)
    parser.add_argument("--vocab", type=Path, default=Config.VOCAB_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells-per-stage", type=int, default=128)
    parser.add_argument("--input-genes", type=int, default=1199)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 mantém o decoder linear do baseline.")
    parser.add_argument("--decoder-batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--variance-weight", type=float, default=0.0, help="Peso da perda de variância por gene no batch.")
    parser.add_argument("--covariance-weight", type=float, default=0.0, help="Peso da perda de covariância em genes variáveis do batch.")
    parser.add_argument("--covariance-genes", type=int, default=128, help="Número de genes de maior variância usados na perda de covariância.")
    postprocessing = parser.add_mutually_exclusive_group()
    postprocessing.add_argument("--snap-to-e95", action="store_true", help="Projeta a saída na célula E9.5 amostrada mais próxima; diagnóstico com vazamento, não submetível.")
    postprocessing.add_argument("--add-validation-residual", action="store_true", help="Soma o resíduo da própria validação; teto diagnóstico com vazamento, não submetível.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction deve estar entre 0 e 1.")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads deve ser positivo.")
    set_seed(args.seed, args.torch_threads)
    e85, e95 = sample_stage(args.e85, args.cells_per_stage, args.seed), sample_stage(args.e95, args.cells_per_stage, args.seed + 1)
    if not e85.var_names.equals(e95.var_names):
        raise ValueError("E8.5 e E9.5 precisam ter genes idênticos e na mesma ordem.")
    expression = sp.vstack([e85.X, e95.X]).tocsr().astype(np.float32)
    stages = np.array(["E8.5"] * e85.n_obs + ["E9.5"] * e95.n_obs)
    train_index, validation_index = split_by_stage(stages, args.validation_fraction, args.seed)
    vocab = json.loads(args.vocab.read_text())
    selected = select_input_genes(expression[train_index], e85.var_names.tolist(), vocab, args.input_genes)
    ids, values, mask = build_sequences(expression, e85.var_names.tolist(), selected, vocab, Config.N_BINS)
    encoder = ScGPTRegressor(len(vocab), Config.D_MODEL, Config.N_HEAD, Config.D_HID, Config.N_LAYERS, Config.DROPOUT)
    load_pretrained_weights(encoder, args.checkpoint)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    latent = extract_embeddings(encoder, ids, values, mask, args.embedding_batch_size)
    if args.hidden_dim < 0 or args.variance_weight < 0 or args.covariance_weight < 0:
        raise ValueError("--hidden-dim, --variance-weight e --covariance-weight não podem ser negativos.")
    covariance_gene_indices = None
    if args.covariance_weight:
        covariance_gene_indices = torch.from_numpy(select_high_variance_gene_indices(expression[train_index], args.covariance_genes))
    decoder = NonnegativeLinearDecoder(latent.shape[1], expression.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(latent[train_index]), torch.from_numpy(expression[train_index].toarray().astype(np.float32))), batch_size=args.decoder_batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    history = []
    for epoch in range(args.epochs):
        decoder.train()
        losses = []
        reconstruction_losses = []
        covariance_losses = []
        for batch_latent, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_prediction = decoder(batch_latent)
            reconstruction_loss = loss_fn(batch_prediction, batch_target)
            loss = reconstruction_loss
            if args.variance_weight:
                loss = loss + args.variance_weight * loss_fn(batch_prediction.var(dim=0, unbiased=False), batch_target.var(dim=0, unbiased=False))
            covariance_loss = torch.zeros((), dtype=batch_prediction.dtype)
            if covariance_gene_indices is not None:
                covariance_loss = normalized_covariance_loss(batch_prediction[:, covariance_gene_indices], batch_target[:, covariance_gene_indices])
                loss = loss + args.covariance_weight * covariance_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            reconstruction_losses.append(float(reconstruction_loss.detach()))
            covariance_losses.append(float(covariance_loss.detach()))
        decoder.eval()
        with torch.no_grad():
            validation_prediction = decoder(torch.from_numpy(latent[validation_index])).numpy()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "train_reconstruction_mse": float(np.mean(reconstruction_losses)), "train_covariance_loss": float(np.mean(covariance_losses)), "validation_mse": float(np.mean((validation_prediction - expression[validation_index].toarray()) ** 2))})
    target_validation = expression[validation_index].toarray().astype(np.float32)
    target_train = expression[train_index].toarray().astype(np.float32)
    raw_metrics = population_metrics(validation_prediction, target_validation, target_train)
    snap_indices = None
    if args.snap_to_e95:
        reference_e95 = expression[e85.n_obs:].toarray().astype(np.float32)
        validation_prediction, snap_indices = snap_to_reference_expression(validation_prediction, reference_e95)
    if args.add_validation_residual:
        validation_prediction = add_oracle_residual(validation_prediction, target_validation)
    metrics = population_metrics(validation_prediction, target_validation, target_train)
    if args.snap_to_e95 or args.add_validation_residual:
        metrics["raw_decoder_metrics"] = raw_metrics
    if args.snap_to_e95:
        metrics["postprocess"] = "nearest_e95_expression_reference_including_validation_LEAKAGE_DIAGNOSTIC"
        metrics["reference_cells"] = int(e95.n_obs)
        metrics["unique_reference_cells_used"] = int(np.unique(snap_indices).size)
    if args.add_validation_residual:
        metrics["postprocess"] = "exact_validation_residual_ORACLE_LEAKAGE_DIAGNOSTIC"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "validation_roundtrip.npz", prediction=validation_prediction.astype(np.float32), decoder_prediction=decoder(torch.from_numpy(latent[validation_index])).detach().numpy().astype(np.float32), target=target_validation, stages=stages[validation_index], snap_indices=snap_indices if snap_indices is not None else np.array([], dtype=np.int64))
    torch.save({"state_dict": decoder.state_dict(), "input_dim": int(latent.shape[1]), "output_dim": int(expression.shape[1]), "selected_genes": selected, "seed": args.seed, "source_status": "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"}, args.output_dir / "decoder.pt")
    config = vars(args) | {"selected_gene_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(), "covariance_gene_sha256": hashlib.sha256(covariance_gene_indices.numpy().tobytes()).hexdigest() if covariance_gene_indices is not None else None, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "n_train": int(len(train_index)), "n_validation": int(len(validation_index)), "gene_token_mapping": "exact_or_uppercase_only", "source_status": "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO", "leakage_diagnostic": bool(args.snap_to_e95 or args.add_validation_residual)}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps({"history": history, "metrics": metrics}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
