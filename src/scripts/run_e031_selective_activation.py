"""E031: selectively activate high-confidence zeros in grouped transitions."""

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
from src.scripts.run_e029_generate import group_expression_delta, normalize_log1p_cp10k
from src.scripts.run_e029_grouped_official import generate, generate_zero_weighted, group_state
from src.scripts.run_m3_known_population import expression_metrics


def prevalence_policy(
    source: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
    groups: int,
    min_target_prevalence: float,
    min_prevalence_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a group/gene activation mask using training prevalence only."""
    mask = np.zeros((groups, source.shape[1]), dtype=bool)
    confidence = np.zeros((groups, source.shape[1]), dtype=np.float32)
    for group in range(groups):
        left = source[source_labels == group]
        right = target[target_labels == group]
        if not len(left) or not len(right):
            continue
        source_prevalence = (left > 0).mean(0)
        target_prevalence = (right > 0).mean(0)
        gain = target_prevalence - source_prevalence
        positive_mean_delta = right.mean(0) > left.mean(0)
        mask[group] = (
            (target_prevalence >= min_target_prevalence)
            & (gain >= min_prevalence_gain)
            & positive_mean_delta
        )
        confidence[group] = gain.astype(np.float32)
    return mask, confidence


def selective_group_expression(
    base: np.ndarray,
    labels: np.ndarray,
    delta: np.ndarray,
    activation_mask: np.ndarray,
    confidence: np.ndarray,
    donor_expression: np.ndarray,
    donor_labels: np.ndarray,
    *,
    expressed_scale: float,
    activation_scale: float,
    max_activations_per_cell: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Adjust expressed genes and copy a capped same-group donor support into zeros."""
    base = np.asarray(base, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    result = base + np.where(base > 0, delta[labels] * np.float32(expressed_scale), 0)
    result = np.maximum(result, 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    pools = {int(group): np.flatnonzero(donor_labels == group) for group in np.unique(donor_labels)}
    activations = np.zeros(len(base), dtype=np.int32)
    fallback_cells = 0
    for row, group in enumerate(labels):
        pool = pools.get(int(group))
        if pool is None or not len(pool):
            fallback_cells += 1
            continue
        donor = donor_expression[int(rng.choice(pool))]
        candidates = np.flatnonzero((base[row] == 0) & activation_mask[group] & (donor > 0))
        if len(candidates) > max_activations_per_cell:
            order = np.argsort(confidence[group, candidates], kind="stable")[-max_activations_per_cell:]
            candidates = candidates[order]
        result[row, candidates] = donor[candidates] * np.float32(activation_scale)
        activations[row] = len(candidates)
    normalized = normalize_log1p_cp10k(result)
    return normalized, {
        "total_activations": int(activations.sum()),
        "mean_activations_per_cell": float(activations.mean()),
        "max_activations_observed": int(activations.max(initial=0)),
        "cells_without_activation": int((activations == 0).sum()),
        "donor_group_fallback_cells": int(fallback_cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e85", type=Path, default=Path("data/E85.h5ad"))
    parser.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    parser.add_argument("--small-checkpoint", type=Path, default=Path("outputs/population_transformer/E029/seed42_k32_n512/small_checkpoint.pt"))
    parser.add_argument("--medium-checkpoint", type=Path, default=Path("outputs/population_transformer/E029/seed42_k32_n512/medium_checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cells", type=int, default=1024)
    parser.add_argument("--validation-cells", type=int, default=256)
    parser.add_argument("--source-pool-cells", type=int, default=2048)
    parser.add_argument("--test-cells", type=int, default=512)
    parser.add_argument("--n-cells", type=int, default=None)
    parser.add_argument("--final-e10", action="store_true")
    parser.add_argument("--final-source-pool-cells", type=int, default=5000)
    parser.add_argument("--input-genes", type=int, default=512)
    parser.add_argument("--landmarks", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.002)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--min-target-prevalence", type=float, default=0.10)
    parser.add_argument("--min-prevalence-gain", type=float, default=0.05)
    parser.add_argument("--expressed-scale", type=float, default=0.25)
    parser.add_argument("--activation-scale", type=float, default=0.125)
    parser.add_argument("--max-activations-per-cell", type=int, default=256)
    parser.add_argument("--zero-change-scales", nargs="+", type=float, default=None)
    parser.add_argument("--models", nargs="+", choices=["group_delta", "small", "medium"],
                        default=["group_delta", "small", "medium"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed, args.threads)
    output_cells = args.n_cells if args.n_cells is not None else args.test_cells
    if not args.final_e10 and output_cells != args.test_cells:
        raise ValueError("n_cells is only distinct from test_cells for --final-e10")
    if "_ex" in args.e85.name or "_ex" in args.e95.name:
        raise ValueError("E031 forbids _ex sources")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    sizes = (args.train_cells, args.validation_cells, args.source_pool_cells)
    e85, genes = read_partition(args.e85, sizes, args.seed)
    e95, target_genes = read_partition(args.e95, (args.train_cells, args.validation_cells, args.test_cells), args.seed + 1)
    if not np.array_equal(genes, target_genes):
        raise ValueError("Gene order mismatch")
    stacked = np.concatenate([e85[0][0], e95[0][0]])
    variance = stacked.var(0)
    selected = np.sort(np.argpartition(variance, -args.input_genes)[-args.input_genes:])
    mean = stacked[:, selected].mean(0)
    scale = stacked[:, selected].std(0) + 1e-4
    normalize = lambda values: ((values[:, selected] - mean) / scale).astype(np.float32)
    encoder = ApproximateKernelPCA(args.landmarks, args.latent_dim, args.gamma, args.ridge, args.seed)
    encoder.fit_embedding(np.concatenate([normalize(e85[0][0]), normalize(e95[0][0])]))
    latent85 = [encoder.transform(normalize(part[0])) for part in e85]
    latent95 = [encoder.transform(normalize(part[0])) for part in e95]
    encoder.fit_decoder([latent85[0], latent95[0]], [e85[0][0], e95[0][0]])
    clusterer = MiniBatchKMeans(args.groups, random_state=args.seed, n_init=5, batch_size=512).fit(
        np.concatenate([latent85[0], latent95[0]])
    )
    states85 = [group_state(values, clusterer) for values in latent85]
    states95 = [group_state(values, clusterer) for values in latent95]
    train_labels85, train_labels95 = states85[0][2], states95[0][2]
    expression_delta = group_expression_delta(e85[0][0], train_labels85, e95[0][0], train_labels95, args.groups)
    activation_mask, confidence = prevalence_policy(
        e85[0][0], train_labels85, e95[0][0], train_labels95, args.groups,
        args.min_target_prevalence, args.min_prevalence_gain,
    )
    group_support = np.zeros((args.groups, e95[0][0].shape[1]), dtype=bool)
    for group in range(args.groups):
        members = e95[0][0][train_labels95 == group]
        if len(members):
            group_support[group] = (members > 0).any(0)

    if args.final_e10:
        if output_cells > args.final_source_pool_cells:
            raise ValueError("n_cells cannot exceed final_source_pool_cells")
        final_parts, final_genes = read_partition(args.e95, (args.final_source_pool_cells, 1, 1), args.seed + 1000)
        if not np.array_equal(genes, final_genes):
            raise ValueError("Gene order mismatch in final source pool")
        source = final_parts[0][0]
        source_latent = encoder.transform(normalize(source))
        input_state, input_features, source_labels = group_state(source_latent, clusterer)
        delta_proportion = states95[0][0]["proportion"] - states85[0][0]["proportion"]
        extrapolated_proportion = np.maximum(input_state["proportion"] + delta_proportion, 0)
        extrapolated_proportion /= extrapolated_proportion.sum()
        group_delta_prediction = {
            "proportion": extrapolated_proportion.astype(np.float32),
            "latent_mean": (input_state["latent_mean"] + states95[0][0]["latent_mean"] - states85[0][0]["latent_mean"]).astype(np.float32),
            "latent_dispersion": np.maximum(input_state["latent_dispersion"] + states95[0][0]["latent_dispersion"] - states85[0][0]["latent_dispersion"], 0).astype(np.float32),
        }
        target = None
    else:
        source = e85[2][0]
        source_latent = latent85[2]
        input_state, input_features, source_labels = states85[2]
        group_delta_prediction = {
            "proportion": states95[0][0]["proportion"].copy(),
            "latent_mean": input_state["latent_mean"] + states95[0][0]["latent_mean"] - states85[0][0]["latent_mean"],
            "latent_dispersion": np.maximum(input_state["latent_dispersion"] + states95[0][0]["latent_dispersion"] - states85[0][0]["latent_dispersion"], 0),
        }
        target = e95[2][0]
    predictions = {"group_delta": group_delta_prediction} if "group_delta" in args.models else {}
    checkpoints = {}
    ids = torch.arange(args.groups, device=device)[None]
    for name, checkpoint_path in (("small", args.small_checkpoint), ("medium", args.medium_checkpoint)):
        if name not in args.models:
            continue
        architecture = ARCHITECTURES[name]
        model = GroupTransitionTransformer(args.latent_dim, args.groups, architecture["model_dim"], architecture["heads"], architecture["layers"], 0.05).to(device)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(input_features[None], device=device), ids)
        predictions[name] = {key: value[0].cpu().numpy().astype(np.float32) for key, value in raw.items()}
        checkpoints[name] = {"epoch": int(saved["epoch"]), "validation_loss": float(saved["validation_loss"])}

    metrics: dict[str, dict] = {}
    contracts = {}
    activation_stats = {}
    for name, prediction in predictions.items():
        if args.zero_change_scales:
            zero_matrices, labels = generate_zero_weighted(
                prediction, input_state, source_latent, source, source_labels, encoder,
                output_cells, args.seed, group_support, args.zero_change_scales,
            )
            variants = [(f"zero_{beta:g}", normalize_log1p_cp10k(matrix))
                        for beta, matrix in zero_matrices.items()]
            if not args.final_e10:
                decoder, _ = generate(prediction, input_state, source_latent, source, source_labels, encoder, output_cells, args.seed)
                variants.insert(0, ("decoder", normalize_log1p_cp10k(decoder)))
        else:
            variants = None
        supported = prediction["proportion"].copy()
        supported[np.bincount(source_labels, minlength=args.groups) == 0] = 0
        supported /= supported.sum()
        rows = weighted_population_without_replacement(supported, source_labels, output_cells, args.seed)
        labels = source_labels[rows]
        rigid = normalize_log1p_cp10k(np.maximum(
            source[rows] + np.where(source[rows] > 0, expression_delta[labels] * np.float32(args.expressed_scale), 0), 0,
        ).astype(np.float32))
        if variants is None:
            selective, stats = selective_group_expression(
                source[rows], labels, expression_delta, activation_mask, confidence,
                e95[0][0], train_labels95, expressed_scale=args.expressed_scale,
                activation_scale=args.activation_scale, max_activations_per_cell=args.max_activations_per_cell,
                seed=args.seed,
            )
            activation_stats[name] = stats
            if args.final_e10:
                variants = (("selective", selective),)
            else:
                decoder, _ = generate(prediction, input_state, source_latent, source, source_labels, encoder, output_cells, args.seed)
                decoder = normalize_log1p_cp10k(decoder)
                variants = (("decoder", decoder), ("rigid", rigid), ("selective", selective))
        for variant, matrix in variants:
            key = f"{name}_{variant}"
            summary = {"zero_fraction": float((matrix == 0).mean()),
                       "unique_cells": int(np.unique(matrix, axis=0).shape[0]),
                       "min": float(matrix.min()), "max": float(matrix.max()),
                       "cp10k_max_abs_error": float(np.max(np.abs(np.expm1(matrix).sum(1) - 10_000)))}
            metrics[key] = summary if target is None else expression_metrics(matrix, target, args.seed) | summary
            output = ad.AnnData(matrix, obs={"group_id": labels.astype(str)}, var={"gene": genes})
            output.var_names = genes
            experiment = "E032" if args.zero_change_scales else "E031"
            output.uns.update(experiment=experiment, model=name, variant=variant, seed=args.seed,
                              sources=["E85.h5ad", "E95.h5ad"], external_ex_used=False,
                              target_stage="E10.5" if args.final_e10 else "E9.5 holdout",
                              target_data_used_for_policy=False, cell_pairing=False)
            contracts[key] = validate_task1_output(output, genes, min_cells=output_cells, max_cells=output_cells)
            filename = f"e10_5_{name}_{variant}_{output_cells}.h5ad" if args.final_e10 else f"{key}.h5ad"
            output.write_h5ad(args.output_dir / filename, compression="gzip")
    eligible = activation_mask.sum(1)
    result = {
        "metrics": metrics,
        "contracts": contracts,
        "activation_stats": activation_stats,
        "policy": {
            "eligible_group_gene_pairs": int(activation_mask.sum()),
            "eligible_genes_union": int(activation_mask.any(0).sum()),
            "eligible_per_group_min_median_max": [int(eligible.min()), float(np.median(eligible)), int(eligible.max())],
        },
        "checkpoints": checkpoints,
        "interpretation": "E10.5 extrapolation without target access" if args.final_e10 else "known E8.5-to-E9.5 transition only; not E10.5 predictiveness",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
