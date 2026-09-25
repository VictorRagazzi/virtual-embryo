"""E021: Transformer residual with expression-anchored population generation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from src.approaches.population_transformer.anchored import (
    anchored_expression, hybrid_gene_programs, weighted_population_without_replacement)
from src.approaches.population_transformer.objectives import multicut_signed_de_loss
from src.approaches.population_transformer.temporal import PopulationTransformer, state_for_source, token_features
from src.m0_contract import validate_task1_output
from src.scripts.run_e018_population_transformer import normalized_target, scales
from src.scripts.run_m3_known_population import expression_metrics


EXTERNAL = ["E65_ex.h5ad", "E675_ex.h5ad", "E70_ex.h5ad", "E725_ex.h5ad", "E75_ex.h5ad",
            "E775_ex.h5ad", "E80_ex.h5ad", "E825_ex.h5ad", "E85_ex.h5ad"]


def seed_all(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def source_time(name: str) -> float:
    digits = Path(name).stem.removesuffix("_ex").removeprefix("E")
    return float(f"{digits[0]}.{digits[1:]}")


def group_expression_means(path: Path, labels: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate full-gene group means without densifying the cell matrix."""
    data = ad.read_h5ad(path, backed="r")
    if data.n_obs != len(labels):
        data.file.close(); raise ValueError(f"Label count mismatch for {path}")
    means = np.zeros((groups, data.n_vars), dtype=np.float32)
    counts = np.bincount(labels, minlength=groups)
    for group in np.flatnonzero(counts):
        block = data.X[np.flatnonzero(labels == group)]
        means[group] = np.asarray(block.mean(axis=0)).ravel().astype(np.float32)
    genes = data.var_names.to_numpy(dtype=str); data.file.close()
    return means, counts, genes


def temporal_changes(means: list[np.ndarray], counts: list[np.ndarray], targets: range) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows, locations = [], []
    for target in targets:
        observed = np.flatnonzero((counts[target] > 0) & (counts[target - 1] > 0))
        for group in observed:
            rows.append(means[target][group] - means[target - 1][group]); locations.append((target, int(group)))
    return np.asarray(rows, dtype=np.float32), locations


def program_targets(means: list[np.ndarray], programs: np.ndarray, target: int) -> np.ndarray:
    return ((means[target] - means[target - 1]) @ programs.T).astype(np.float32)


def train_model(states, times, means, counts, programs, program_scores, program_y,
                train_targets, holdout_target, args, device):
    center_mean, center_scale, dispersion_scale = scales([states[i] for i in range(holdout_target)])
    score_train = np.concatenate([program_scores[i] for i in range(holdout_target)])
    score_mean = score_train.mean(0); score_scale = score_train.std(0) + 1e-6
    delta_train = np.concatenate([program_y[i] for i in train_targets])
    delta_mean = delta_train.mean(0); delta_scale = delta_train.std(0) + 1e-6
    def augmented(target):
        features, ids = token_features(states[target-2:target], times[target-2:target], times[target],
                                       center_mean, center_scale, dispersion_scale)
        scores = np.concatenate([program_scores[target-2], program_scores[target-1]])
        return np.concatenate([features, (scores - score_mean) / score_scale], axis=1).astype(np.float32), ids
    program_tensor = torch.tensor(programs, device=device)
    def gene_delta_target(target):
        previous_weight = counts[target - 1] / counts[target - 1].sum()
        target_weight = counts[target] / counts[target].sum()
        reference = (previous_weight[:, None] * means[target - 1]).sum(0)
        truth = (target_weight[:, None] * means[target]).sum(0)
        return (torch.tensor(means[target - 1], device=device), torch.tensor(reference, device=device),
                torch.tensor(truth - reference, device=device))
    examples = []
    for target in train_targets:
        features, ids = augmented(target)
        examples.append((features, ids, normalized_target(states[target], center_mean, center_scale,
                                                           dispersion_scale, device),
                         torch.tensor((program_y[target] - delta_mean) / delta_scale, device=device),
                         *gene_delta_target(target)))
    hold_features, hold_ids = augmented(holdout_target)
    hold_target = normalized_target(states[holdout_target], center_mean, center_scale, dispersion_scale, device)
    hold_program = torch.tensor((program_y[holdout_target] - delta_mean) / delta_scale, device=device)
    hold_previous_means, hold_reference, hold_gene_delta = gene_delta_target(holdout_target)
    model = PopulationTransformer(hold_features.shape[1], states[0]["latent_mean"].shape[1], len(states[0]["proportion"]),
        args.model_dim, args.heads, args.layers, args.dropout, args.n_programs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, bad, history = float("inf"), 0, []
    checkpoint = args.output_dir / "checkpoint.pt"
    for epoch in range(args.epochs):
        model.train(); losses = []
        for features, ids, target, target_programs, previous_means, reference, target_gene_delta in examples:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.tensor(features[None], device=device), torch.tensor(ids[None], device=device))
            mask = target["proportion"] > 0
            structural = ((prediction["proportion"] - target["proportion"]) ** 2).mean()
            structural += ((prediction["latent_mean"][:, mask] - target["latent_mean"][mask]) ** 2).mean()
            structural += ((prediction["latent_dispersion"][:, mask] - target["latent_dispersion"][mask]) ** 2).mean()
            program = ((prediction["program_delta"][:, mask] - target_programs[mask]) ** 2).mean()
            loss = args.structural_weight * structural + args.program_weight * program
            if args.de_weight:
                actual_coefficients = prediction["program_delta"][0] * torch.tensor(delta_scale, device=device) + torch.tensor(delta_mean, device=device)
                group_delta = actual_coefficients @ program_tensor
                future = (prediction["proportion"][0, :, None] * (previous_means + group_delta)).sum(0)
                loss = loss + args.de_weight * multicut_signed_de_loss(
                    future - reference, target_gene_delta, args.de_cuts, args.de_temperature)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(hold_features[None], device=device), torch.tensor(hold_ids[None], device=device))
            mask = hold_target["proportion"] > 0
            validation = ((pred["proportion"] - hold_target["proportion"]) ** 2).mean()
            validation += ((pred["latent_mean"][:, mask] - hold_target["latent_mean"][mask]) ** 2).mean()
            validation += ((pred["latent_dispersion"][:, mask] - hold_target["latent_dispersion"][mask]) ** 2).mean()
            validation *= args.structural_weight
            validation += args.program_weight * ((pred["program_delta"][:, mask] - hold_program[mask]) ** 2).mean()
            if args.de_weight:
                actual_coefficients = pred["program_delta"][0] * torch.tensor(delta_scale, device=device) + torch.tensor(delta_mean, device=device)
                group_delta = actual_coefficients @ program_tensor
                future = (pred["proportion"][0, :, None] * (hold_previous_means + group_delta)).sum(0)
                validation += args.de_weight * multicut_signed_de_loss(
                    future - hold_reference, hold_gene_delta, args.de_cuts, args.de_temperature)
            value = float(validation)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": value})
        if value < best:
            best, bad = value, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_loss": value}, checkpoint)
        else:
            bad += 1
            if bad >= args.patience: break
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["model"])
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    program_norm = (score_mean, score_scale, delta_mean, delta_scale)
    return model, (center_mean, center_scale, dispersion_scale), program_norm, hold_features, hold_ids, saved


def read_rows(path: Path, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r"); unique, inverse = np.unique(rows, return_inverse=True)
    matrix = data.X[unique]; dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    genes = data.var_names.to_numpy(dtype=str)
    labels = data.obs["celltype"].to_numpy(dtype=str)[unique][inverse] if "celltype" in data.obs else np.repeat("NA", len(rows))
    data.file.close(); return dense[inverse].astype(np.float32), genes, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("outputs/population_transformer/cache/embeddings.npz"))
    p.add_argument("--tokens", type=Path, default=Path("outputs/population_transformer/cache/tokens_k50.npz"))
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-programs", type=int, default=32); p.add_argument("--n-cells", type=int, default=512)
    p.add_argument("--de-programs", type=int, default=0)
    p.add_argument("--experiment", default="E021")
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--patience", type=int, default=25)
    p.add_argument("--model-dim", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4); p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--program-weight", type=float, default=1.); p.add_argument("--scales", nargs="+", type=float, default=[.25, .5, 1.])
    p.add_argument("--structural-weight", type=float, default=1.)
    p.add_argument("--de-weight", type=float, default=0.)
    p.add_argument("--de-cuts", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    p.add_argument("--de-temperature", type=float, default=.1)
    p.add_argument("--proportion-scale", type=float, default=.25,
                   help="Residual shrinkage applied to Transformer composition changes.")
    p.add_argument("--allow-activation", action="store_true", help="Allow historically supported activation of source zeros.")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--program-seed", type=int, default=42)
    p.add_argument("--evaluation-seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    cache, tokens = np.load(args.cache), np.load(args.tokens); groups = tokens["centers"].shape[0]
    names = EXTERNAL; states = [state_for_source(cache, tokens, name) for name in names]
    times = [source_time(name) for name in names]
    means, counts, genes = [], [], None
    for name in names:
        labels = tokens["labels"][cache["sources"] == name]
        current_mean, current_count, current_genes = group_expression_means(args.data_dir / name, labels, groups)
        if genes is not None and not np.array_equal(genes, current_genes): raise ValueError("Gene order mismatch")
        means.append(current_mean); counts.append(current_count); genes = current_genes
    train_targets = range(2, len(names) - 1); changes, _ = temporal_changes(means, counts, train_targets)
    pseudobulks = [(m * (c / c.sum())[:, None]).sum(0) for m, c in zip(means, counts)]
    pb_changes = np.asarray([pseudobulks[t] - pseudobulks[t-1] for t in train_targets], dtype=np.float32)
    programs = hybrid_gene_programs(changes, pb_changes, args.n_programs - args.de_programs,
                                   args.de_programs, args.program_seed)
    targets = [np.zeros((groups, args.n_programs), np.float32) for _ in names]
    for target in range(1, len(names)): targets[target] = program_targets(means, programs, target)
    program_scores = [(value @ programs.T).astype(np.float32) for value in means]
    model, norm, program_norm, features, ids, selected = train_model(
        states, times, means, counts, programs, program_scores, targets, train_targets, len(names)-1, args, device)
    model.eval()
    with torch.no_grad(): raw = model(torch.tensor(features[None], device=device), torch.tensor(ids[None], device=device))
    predicted_proportion = raw["proportion"][0].cpu().numpy()
    proportion = states[-2]["proportion"] + args.proportion_scale * (predicted_proportion - states[-2]["proportion"])
    proportion = np.maximum(proportion, 0); proportion = proportion / proportion.sum()
    coefficients = raw["program_delta"][0].cpu().numpy() * program_norm[3] + program_norm[2]
    previous_mask = cache["sources"] == names[-2]; previous_labels = tokens["labels"][previous_mask]
    selected_rows = weighted_population_without_replacement(proportion, previous_labels, args.n_cells, args.evaluation_seed)
    generated_groups = previous_labels[selected_rows]; fallback = 0
    base, base_genes, base_celltypes = read_rows(args.data_dir / names[-2], selected_rows)
    reference_rows = weighted_population_without_replacement(
        states[-2]["proportion"], previous_labels, args.n_cells, args.evaluation_seed)
    reference_expression, reference_genes, reference_celltypes = read_rows(args.data_dir / names[-2], reference_rows)
    rng = np.random.default_rng(args.evaluation_seed); target_data = ad.read_h5ad(args.data_dir / names[-1], backed="r")
    target_rows = np.sort(rng.choice(target_data.n_obs, args.n_cells, replace=False)); target_data.file.close()
    target, target_genes, target_celltypes = read_rows(args.data_dir / names[-1], target_rows)
    if not (np.array_equal(base_genes, target_genes) and np.array_equal(reference_genes, target_genes)
            and np.array_equal(genes, target_genes)):
        raise ValueError("Gene order mismatch")
    activation = np.zeros((groups, len(genes)), dtype=bool)
    for target_index in train_targets:
        activation |= (means[target_index - 1] <= 0) & (means[target_index] > 0)
    variants = {"copy_last": reference_expression}
    for scale in args.scales:
        variants[f"anchored_{scale:g}"] = anchored_expression(base, generated_groups, coefficients, programs,
            scale=scale, activation_mask=activation if args.allow_activation else None)
    metrics = {}
    for name, matrix in variants.items():
        metrics[name] = expression_metrics(matrix, target, args.evaluation_seed) | {
            "zero_fraction": float((matrix == 0).mean()), "unique_cells": int(np.unique(matrix, axis=0).shape[0])}
        output = ad.AnnData(matrix, obs={"group_id": generated_groups.astype(str)}, var={"gene": genes}); output.var_names = genes
        output.uns.update(experiment=args.experiment, seed=args.seed, variant=name, pseudo_target="E85_ex.h5ad")
        validate_task1_output(output, genes, min_cells=args.n_cells, max_cells=args.n_cells)
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    reference = ad.AnnData(reference_expression, obs={"celltype": reference_celltypes}, var={"gene": genes}); reference.var_names = genes
    reference.write_h5ad(args.output_dir / "reference.h5ad", compression="gzip")
    truth = ad.AnnData(target, obs={"celltype": target_celltypes}, var={"gene": genes}); truth.var_names = genes
    truth.write_h5ad(args.output_dir / "target.h5ad", compression="gzip")
    result = {"metrics": metrics, "best_epoch": selected["epoch"], "validation_loss": selected["validation_loss"],
              "fallback_cells": fallback, "device": str(device), "programs": args.n_programs,
              "note": "Local proxies only; run veckit on saved h5ad files for official-aligned metrics."}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    np.savez_compressed(args.output_dir / "programs.npz", programs=programs, activation_mask=activation,
                        score_mean=program_norm[0], score_scale=program_norm[1],
                        delta_mean=program_norm[2], delta_scale=program_norm[3])
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
