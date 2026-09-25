"""M3: reconstrói uma população conhecida por centros + resíduos empíricos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from src.scripts.run_scgpt_roundtrip import NonnegativeLinearDecoder


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def sample_population(latent: np.ndarray, stages: np.ndarray, labels: np.ndarray, stage: str, n_cells: int,
                      seed: int, use_residuals: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Amostra grupos reais e resíduos do mesmo `(estágio, grupo)`."""
    rng = np.random.default_rng(seed)
    stage_mask = stages == stage
    if not stage_mask.any():
        raise ValueError(f"Estágio ausente: {stage}")
    k = int(labels.max()) + 1
    counts = np.bincount(labels[stage_mask], minlength=k)
    requested = rng.multinomial(n_cells, counts / counts.sum())
    generated, generated_labels = [], []
    fallbacks = 0
    for group, amount in enumerate(requested):
        if not amount:
            continue
        pool = latent[stage_mask & (labels == group)]
        if not len(pool):
            nonempty = np.flatnonzero(counts)
            group = int(nonempty[np.argmin(abs(nonempty - group))])
            pool = latent[stage_mask & (labels == group)]
            fallbacks += amount
        center = pool.mean(axis=0)
        if use_residuals:
            residuals = pool - center
            chunk = center + residuals[rng.integers(0, len(residuals), amount)]
        else:
            chunk = np.repeat(center[None], amount, axis=0)
        generated.append(chunk.astype(np.float32)); generated_labels.extend([group] * amount)
    order = rng.permutation(n_cells)
    return np.concatenate(generated)[order], np.asarray(generated_labels, dtype=np.int32)[order], fallbacks


def decode(decoder: NonnegativeLinearDecoder, latent: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    decoder.to(device).eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            chunks.append(decoder(torch.from_numpy(latent[start:start + batch_size]).to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def rff_mmd(x: np.ndarray, y: np.ndarray, features: int, seed: int) -> float:
    """MMD RBF aproximada em memória O((n+d)*features), sem matriz kernel completa."""
    rng = np.random.default_rng(seed)
    joined = np.vstack([x, y])
    probe = joined[rng.choice(len(joined), min(256, len(joined)), replace=False)]
    distances = np.sqrt(np.maximum(((probe[:128, None] - probe[None, 128:]) ** 2).sum(axis=2), 0)).ravel()
    sigma = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    weights = rng.normal(size=(x.shape[1], features)).astype(np.float32) / sigma
    bias = rng.uniform(0, 2 * np.pi, features).astype(np.float32)
    scale = np.sqrt(2 / features)
    phi_x = scale * np.cos(x @ weights + bias)
    phi_y = scale * np.cos(y @ weights + bias)
    return float(np.square(phi_x.mean(0) - phi_y.mean(0)).sum())


def expression_metrics(prediction: np.ndarray, target: np.ndarray, seed: int) -> dict[str, float]:
    variance = target.var(axis=0)
    top = np.argsort(-variance, kind="stable")[:256]
    cov_top = top[:64]
    pred_cov, target_cov = np.cov(prediction[:, cov_top], rowvar=False), np.cov(target[:, cov_top], rowvar=False)
    upper = np.triu_indices(len(cov_top), 1)
    return {"pseudobulk_pearson": float(np.corrcoef(prediction.mean(0), target.mean(0))[0, 1]),
            "pseudobulk_relative_error": float(np.linalg.norm(prediction.mean(0) - target.mean(0)) / np.linalg.norm(target.mean(0))),
            "variance_ratio": float(prediction.var(0).mean() / (target.var(0).mean() + 1e-12)),
            "sampled_covariance_pearson": float(np.corrcoef(pred_cov[upper], target_cov[upper])[0, 1]),
            "rff_mmd_top256": rff_mmd(prediction[:, top], target[:, top], 256, seed)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True); p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--stage", default="E9.5")
    p.add_argument("--n-cells", type=int, default=512); p.add_argument("--sensitivity-cells", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--decoder-batch-size", type=int, default=32); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed, args.threads); args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, tokens = np.load(args.cache), np.load(args.tokens)
    latent, stages, labels = cache["latent"], cache["stages"], tokens["labels"]
    decoder_data = torch.load(args.decoder, map_location="cpu", weights_only=False)
    if decoder_data["input_dim"] != latent.shape[1]: raise ValueError("Dimensão latente incompatível.")
    if decoder_data["selected_genes"] != cache["selected_genes"].tolist(): raise ValueError("Tokenização do decoder e cache incompatíveis.")
    decoder = NonnegativeLinearDecoder(decoder_data["input_dim"], decoder_data["output_dim"])
    decoder.load_state_dict(decoder_data["state_dict"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    residual_latent, residual_groups, fallback = sample_population(latent, stages, labels, args.stage, args.n_cells, args.seed, True)
    centroid_latent, centroid_groups, _ = sample_population(latent, stages, labels, args.stage, args.n_cells, args.seed, False)
    residual_expression = decode(decoder, residual_latent, args.decoder_batch_size, device)
    centroid_expression = decode(decoder, centroid_latent, args.decoder_batch_size, device)
    target = ad.read_h5ad(args.target, backed="r")
    rng = np.random.default_rng(args.seed); target_indices = np.sort(rng.choice(target.n_obs, args.n_cells, replace=False))
    target_matrix = target.X[target_indices]
    target_matrix = target_matrix.toarray().astype(np.float32) if sp.issparse(target_matrix) else np.asarray(target_matrix, dtype=np.float32)
    genes = target.var_names.copy(); target.file.close()
    metrics = {"residual": expression_metrics(residual_expression, target_matrix, args.seed),
               "centroid": expression_metrics(centroid_expression, target_matrix, args.seed), "fallback_cells": fallback,
               "unique_residual_cells": int(np.unique(residual_expression, axis=0).shape[0]),
               "unique_centroid_cells": int(np.unique(centroid_expression, axis=0).shape[0]), "sensitivity": {}}
    target_proportions = np.bincount(labels[stages == args.stage], minlength=int(labels.max()) + 1); target_proportions = target_proportions / target_proportions.sum()
    for n in args.sensitivity_cells:
        sampled, groups, fallbacks = sample_population(latent, stages, labels, args.stage, n, args.seed, True)
        observed = np.bincount(groups, minlength=len(target_proportions)) / n
        metrics["sensitivity"][str(n)] = {"composition_l1": float(np.abs(observed - target_proportions).sum()),
            "latent_variance_ratio": float(sampled.var(0).mean() / latent[stages == args.stage].var(0).mean()), "fallback_cells": fallbacks}
    for name, matrix, groups in [("residual", residual_expression, residual_groups), ("centroid", centroid_expression, centroid_groups)]:
        output = ad.AnnData(X=matrix, obs={"group_id": groups.astype(str), "generator": name}, var={"gene": genes.astype(str)})
        output.var_names = genes; output.uns["seed"] = args.seed; output.uns["source_status"] = "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"
        output.write_h5ad(args.output_dir / f"e95_{name}_{args.n_cells}.h5ad", compression="gzip")
    config = vars(args) | {"device_used": str(device), "source_status": "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
