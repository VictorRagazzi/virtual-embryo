"""Retrain E021 on all allowed transitions and generate anchored E10.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from src.approaches.population_transformer.anchored import (
    anchored_expression, fit_gene_programs, weighted_population_without_replacement)
from src.approaches.population_transformer.objectives import multicut_signed_de_loss
from src.approaches.population_transformer.temporal import PopulationTransformer, state_for_source, token_features
from src.m0_contract import validate_task1_output
from src.scripts.run_e018_population_transformer import normalized_target, scales
from src.scripts.run_e021_anchored import (EXTERNAL, group_expression_means, read_rows, seed_all, source_time)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("outputs/population_transformer/cache/embeddings.npz"))
    p.add_argument("--tokens", type=Path, default=Path("outputs/population_transformer/cache/tokens_k50.npz"))
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-cells", type=int, default=2500); p.add_argument("--n-programs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--model-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.05); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--program-weight", type=float, default=1.)
    p.add_argument("--de-weight", type=float, default=0.)
    p.add_argument("--de-cuts", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    p.add_argument("--de-temperature", type=float, default=.1)
    p.add_argument("--expression-scale", type=float, default=.25); p.add_argument("--proportion-scale", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    cache, tokens = np.load(args.cache), np.load(args.tokens); groups = tokens["centers"].shape[0]
    names = EXTERNAL + ["E85.h5ad", "E95.h5ad"]
    states = [state_for_source(cache, tokens, name) for name in names]
    times = [source_time(name) for name in names]
    means, counts, genes = [], [], None
    for name in names:
        labels = tokens["labels"][cache["sources"] == name]
        mean, count, current_genes = group_expression_means(args.data_dir / name, labels, groups)
        if genes is not None and not np.array_equal(genes, current_genes): raise ValueError("Gene order mismatch")
        means.append(mean); counts.append(count); genes = current_genes

    # External consecutive transitions plus the official E8.5 -> E9.5 change.
    transition_pairs = [(i - 1, i) for i in range(2, len(EXTERNAL))]
    transition_pairs.append((len(EXTERNAL), len(EXTERNAL) + 1))
    change_rows = []
    for previous, target in transition_pairs:
        observed = np.flatnonzero((counts[previous] > 0) & (counts[target] > 0))
        change_rows.extend(means[target][observed] - means[previous][observed])
    _, programs = fit_gene_programs(np.asarray(change_rows, np.float32), args.n_programs, args.seed)
    scores = [(mean @ programs.T).astype(np.float32) for mean in means]
    deltas = {(previous, target): ((means[target] - means[previous]) @ programs.T).astype(np.float32)
              for previous, target in transition_pairs}

    # Examples: all external triples and E8.25_ex, official E8.5 -> official E9.5.
    specs = [(i - 2, i - 1, i) for i in range(2, len(EXTERNAL))]
    specs.append((len(EXTERNAL) - 2, len(EXTERNAL), len(EXTERNAL) + 1))
    center_mean, center_scale, dispersion_scale = scales(states)
    score_values = np.concatenate(scores[:-1]); score_mean = score_values.mean(0); score_scale = score_values.std(0) + 1e-6
    delta_values = np.concatenate([deltas[(previous, target)] for _, previous, target in specs])
    delta_mean = delta_values.mean(0); delta_scale = delta_values.std(0) + 1e-6

    examples = []
    for first, previous, target in specs:
        features, ids = token_features([states[first], states[previous]], [times[first], times[previous]], times[target],
                                        center_mean, center_scale, dispersion_scale)
        program_input = np.concatenate([scores[first], scores[previous]])
        features = np.concatenate([features, (program_input - score_mean) / score_scale], axis=1).astype(np.float32)
        target_state = normalized_target(states[target], center_mean, center_scale, dispersion_scale, device)
        target_program = torch.tensor((deltas[(previous, target)] - delta_mean) / delta_scale, device=device)
        previous_weight = counts[previous] / counts[previous].sum(); target_weight = counts[target] / counts[target].sum()
        reference = (previous_weight[:, None] * means[previous]).sum(0)
        truth = (target_weight[:, None] * means[target]).sum(0)
        examples.append((features, ids, target_state, target_program,
                         torch.tensor(means[previous], device=device), torch.tensor(reference, device=device),
                         torch.tensor(truth - reference, device=device)))

    model = PopulationTransformer(examples[0][0].shape[1], states[0]["latent_mean"].shape[1], groups,
        args.model_dim, args.heads, args.layers, args.dropout, args.n_programs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train(); epoch_losses = []
        for features, ids, target, target_program, previous_means, reference, target_gene_delta in examples:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.tensor(features[None], device=device), torch.tensor(ids[None], device=device))
            mask = target["proportion"] > 0
            loss = ((prediction["proportion"] - target["proportion"]) ** 2).mean()
            loss += ((prediction["latent_mean"][:, mask] - target["latent_mean"][mask]) ** 2).mean()
            loss += ((prediction["latent_dispersion"][:, mask] - target["latent_dispersion"][mask]) ** 2).mean()
            loss += args.program_weight * ((prediction["program_delta"][:, mask] - target_program[mask]) ** 2).mean()
            if args.de_weight:
                coefficients = prediction["program_delta"][0] * torch.tensor(delta_scale, device=device) + torch.tensor(delta_mean, device=device)
                group_delta = coefficients @ torch.tensor(programs, device=device)
                future = (prediction["proportion"][0, :, None] * (previous_means + group_delta)).sum(0)
                loss += args.de_weight * multicut_signed_de_loss(
                    future - reference, target_gene_delta, args.de_cuts, args.de_temperature)
            loss.backward(); optimizer.step(); epoch_losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses))})

    first, previous = len(EXTERNAL), len(EXTERNAL) + 1
    features, ids = token_features([states[first], states[previous]], [8.5, 9.5], 10.5,
                                    center_mean, center_scale, dispersion_scale)
    program_input = np.concatenate([scores[first], scores[previous]])
    features = np.concatenate([features, (program_input - score_mean) / score_scale], axis=1).astype(np.float32)
    model.eval()
    with torch.no_grad(): prediction = model(torch.tensor(features[None], device=device), torch.tensor(ids[None], device=device))
    predicted_proportion = prediction["proportion"][0].cpu().numpy()
    proportion = states[previous]["proportion"] + args.proportion_scale * (predicted_proportion - states[previous]["proportion"])
    proportion = np.maximum(proportion, 0); proportion /= proportion.sum()
    coefficients = prediction["program_delta"][0].cpu().numpy() * delta_scale + delta_mean
    source_mask = cache["sources"] == "E95.h5ad"; source_groups = tokens["labels"][source_mask]
    selected = weighted_population_without_replacement(proportion, source_groups, args.n_cells, args.seed)
    generated_groups = source_groups[selected]
    base, output_genes, _ = read_rows(args.data_dir / "E95.h5ad", selected)
    expression = anchored_expression(base, generated_groups, coefficients, programs, scale=args.expression_scale)
    output = ad.AnnData(expression, obs={"group_id": generated_groups.astype(str)}, var={"gene": output_genes})
    output.var_names = output_genes
    output.obs_names = [f"E021_{index:04d}" for index in range(args.n_cells)]
    output.uns.update(experiment="E021", seed=args.seed, target_stage="E10.5", target_data_used=False,
                      method="expression_anchored_program_transformer", expression_scale=args.expression_scale,
                      proportion_scale=args.proportion_scale, encoder="scGPT M001")
    contract = validate_task1_output(output, output_genes, min_cells=args.n_cells, max_cells=args.n_cells)
    output_path = args.output_dir / "e10_5_anchored_2500.h5ad"; output.write_h5ad(output_path, compression="gzip")
    torch.save({"model": model.state_dict(), "programs": programs, "score_mean": score_mean,
                "score_scale": score_scale, "delta_mean": delta_mean, "delta_scale": delta_scale,
                "config": vars(args)}, args.output_dir / "checkpoint.pt")
    metrics = {"contract": contract, "unique_cells": int(np.unique(expression, axis=0).shape[0]),
               "zero_fraction": float((expression == 0).mean()), "history": history,
               "output": str(output_path), "source_cells_reused": 0}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__": main()
