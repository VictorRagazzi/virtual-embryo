"""Generate 2,500-cell E10.5 candidates from the grouped official-only E029 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from src.approaches.population_transformer.anchored import weighted_population_without_replacement
from src.approaches.population_transformer.nonlinear_pca import ApproximateKernelPCA, GroupTransitionTransformer
from src.m0_contract import validate_task1_output
from src.scripts.run_e028_official_only import ARCHITECTURES, read_partition, seed_all
from src.scripts.run_e029_grouped_official import generate, group_state


def normalize_log1p_cp10k(matrix: np.ndarray) -> np.ndarray:
    """Restore the expected log1p(CP10k) library scale after decoding."""
    values = np.asarray(matrix, dtype=np.float32)
    counts = np.expm1(np.clip(values, 0, np.float32(np.log1p(1e6))))
    totals = counts.sum(axis=1, keepdims=True)
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise ValueError("Decoded population has invalid library sizes")
    return np.log1p(counts * (np.float32(10_000) / totals)).astype(np.float32)


def group_expression_delta(source: np.ndarray, source_labels: np.ndarray, target: np.ndarray,
                           target_labels: np.ndarray, groups: int) -> np.ndarray:
    """Observed per-group expression change, with zero for unsupported groups."""
    delta = np.zeros((groups, source.shape[1]), dtype=np.float32)
    for group in range(groups):
        left, right = source[source_labels == group], target[target_labels == group]
        if len(left) and len(right): delta[group] = right.mean(0) - left.mean(0)
    return delta


def anchored_group_expression(base: np.ndarray, labels: np.ndarray, delta: np.ndarray, scale: float) -> np.ndarray:
    """Apply group change only on expressed genes, preserving the anchor sparsity pattern."""
    shifted = base + np.where(base > 0, delta[labels] * np.float32(scale), 0)
    return normalize_log1p_cp10k(np.maximum(shifted, 0).astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e85", type=Path, default=Path("data/E85.h5ad")); p.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    p.add_argument("--small-checkpoint", type=Path, default=Path("outputs/population_transformer/E029/seed42_k32_n512/small_checkpoint.pt"))
    p.add_argument("--medium-checkpoint", type=Path, default=Path("outputs/population_transformer/E029/seed42_k32_n512/medium_checkpoint.pt"))
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--n-cells", type=int, default=2500)
    p.add_argument("--source-pool-cells", type=int, default=5000); p.add_argument("--train-cells", type=int, default=1024)
    p.add_argument("--validation-cells", type=int, default=256); p.add_argument("--input-genes", type=int, default=512)
    p.add_argument("--landmarks", type=int, default=256); p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--groups", type=int, default=32); p.add_argument("--gamma", type=float, default=.002)
    p.add_argument("--ridge", type=float, default=.01); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchored-scales", nargs="+", type=float, default=[.125, .25])
    p.add_argument("--threads", type=int, default=4); p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    if args.n_cells > args.source_pool_cells: raise ValueError("n_cells cannot exceed source_pool_cells")
    if "_ex" in args.e85.name or "_ex" in args.e95.name: raise ValueError("E029 generation forbids _ex sources")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    # Rebuild the exact E029 training representation. Partition sizes and seeds match E029.
    e85, genes = read_partition(args.e85, (args.train_cells, args.validation_cells, 2048), args.seed)
    e95, target_genes = read_partition(args.e95, (args.train_cells, args.validation_cells, 512), args.seed + 1)
    if not np.array_equal(genes, target_genes): raise ValueError("Gene order mismatch")
    stacked = np.concatenate([e85[0][0], e95[0][0]]); variance = stacked.var(0)
    selected = np.sort(np.argpartition(variance, -args.input_genes)[-args.input_genes:])
    mean = stacked[:, selected].mean(0); scale = stacked[:, selected].std(0) + 1e-4
    normalize = lambda x: ((x[:, selected] - mean) / scale).astype(np.float32)
    encoder = ApproximateKernelPCA(args.landmarks, args.latent_dim, args.gamma, args.ridge, args.seed)
    encoder.fit_embedding(np.concatenate([normalize(e85[0][0]), normalize(e95[0][0])]))
    train_latent85 = encoder.transform(normalize(e85[0][0])); train_latent95 = encoder.transform(normalize(e95[0][0]))
    encoder.fit_decoder([train_latent85, train_latent95], [e85[0][0], e95[0][0]])
    clusterer = MiniBatchKMeans(args.groups, random_state=args.seed, n_init=5, batch_size=512).fit(
        np.concatenate([train_latent85, train_latent95]))
    train85, _, train_labels85 = group_state(train_latent85, clusterer)
    train95, _, train_labels95 = group_state(train_latent95, clusterer)
    expression_delta = group_expression_delta(e85[0][0], train_labels85, e95[0][0], train_labels95, args.groups)

    # E9.5 is the last observed population and supplies unique anchors for E10.5.
    pool_parts, pool_genes = read_partition(args.e95, (args.source_pool_cells, 1, 1), args.seed + 1000)
    if not np.array_equal(genes, pool_genes): raise ValueError("Gene order mismatch")
    source_expression = pool_parts[0][0]
    source_latent = encoder.transform(normalize(source_expression))
    source_state, source_features, source_labels = group_state(source_latent, clusterer)

    delta_proportion = train95["proportion"] - train85["proportion"]
    extrapolated_proportion = np.maximum(source_state["proportion"] + delta_proportion, 0)
    extrapolated_proportion /= extrapolated_proportion.sum()
    predictions = {"group_delta": {
        "proportion": extrapolated_proportion.astype(np.float32),
        "latent_mean": (source_state["latent_mean"] + train95["latent_mean"] - train85["latent_mean"]).astype(np.float32),
        "latent_dispersion": np.maximum(source_state["latent_dispersion"] + train95["latent_dispersion"] - train85["latent_dispersion"], 0).astype(np.float32)}}

    saved_models = {}
    for model_name, checkpoint_path in (("small", args.small_checkpoint), ("medium", args.medium_checkpoint)):
        architecture = ARCHITECTURES[model_name]
        model = GroupTransitionTransformer(args.latent_dim, args.groups, architecture["model_dim"], architecture["heads"],
                                           architecture["layers"], .05).to(device)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"]); model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(source_features[None], device=device),
                        torch.arange(args.groups, device=device)[None])
        predictions[model_name] = {key: value[0].cpu().numpy().astype(np.float32) for key, value in raw.items()}
        saved_models[model_name] = {"epoch": int(saved["epoch"]), "validation_loss": float(saved["validation_loss"])}

    contracts = {}; summaries = {}
    for name, prediction in predictions.items():
        matrix, groups = generate(prediction, source_state, source_latent, source_expression, source_labels,
                                  encoder, args.n_cells, args.seed)
        matrix = normalize_log1p_cp10k(matrix)
        output = ad.AnnData(matrix, obs={"group_id": groups.astype(str)}, var={"gene": genes}); output.var_names = genes
        output.uns.update(experiment="E029", variant=name, target_stage="E10.5", seed=args.seed,
                          sources=["E85.h5ad", "E95.h5ad"], external_ex_used=False, target_data_used=False,
                          group_definition="joint_train_E85_E95_KMeans", cell_pairing=False,
                          generation="E95 anchors plus extrapolated E85-to-E95 group transition")
        contracts[name] = validate_task1_output(output, genes, min_cells=args.n_cells, max_cells=args.n_cells)
        output.write_h5ad(args.output_dir / f"e10_5_{name}_2500.h5ad", compression="gzip")
        summaries[name] = {"unique_cells": int(np.unique(matrix, axis=0).shape[0]),
                           "zero_fraction": float((matrix == 0).mean()), "min": float(matrix.min()),
                           "max": float(matrix.max()), "groups_used": int(len(np.unique(groups))),
                           "cp10k_max_abs_error": float(np.max(np.abs(np.expm1(matrix).sum(1) - 10_000)))}
        supported = prediction["proportion"].copy()
        supported[np.bincount(source_labels, minlength=len(supported)) == 0] = 0; supported /= supported.sum()
        selected_rows = weighted_population_without_replacement(supported, source_labels, args.n_cells, args.seed)
        selected_groups = source_labels[selected_rows]
        for anchored_scale in args.anchored_scales:
            variant = f"{name}_anchored_{anchored_scale:g}"
            anchored = anchored_group_expression(source_expression[selected_rows], selected_groups,
                                                 expression_delta, anchored_scale)
            output = ad.AnnData(anchored, obs={"group_id": selected_groups.astype(str)}, var={"gene": genes}); output.var_names = genes
            output.uns.update(experiment="E030", variant=variant, target_stage="E10.5", seed=args.seed,
                              sources=["E85.h5ad", "E95.h5ad"], external_ex_used=False, target_data_used=False,
                              generation="E95 expression anchors plus sparse per-group E85-to-E95 gene delta",
                              expression_scale=anchored_scale, cell_pairing=False)
            contracts[variant] = validate_task1_output(output, genes, min_cells=args.n_cells, max_cells=args.n_cells)
            output.write_h5ad(args.output_dir / f"e10_5_{variant}_2500.h5ad", compression="gzip")
            summaries[variant] = {"unique_cells": int(np.unique(anchored, axis=0).shape[0]),
                                  "zero_fraction": float((anchored == 0).mean()), "min": float(anchored.min()),
                                  "max": float(anchored.max()), "groups_used": int(len(np.unique(selected_groups))),
                                  "cp10k_max_abs_error": float(np.max(np.abs(np.expm1(anchored).sum(1) - 10_000)))}
    result = {"contracts": contracts, "summaries": summaries, "checkpoints": saved_models,
              "source_pool_cells": args.source_pool_cells, "target_data_used": False}
    (args.output_dir / "generation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
