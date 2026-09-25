"""E019: pipeline Transformer → latentes → três variantes E10.5 exploratórias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.approaches.population_transformer.temporal import PopulationTransformer, state_for_source, token_features
from src.m0_contract import validate_task1_output
from src.scripts.run_e016_expression_residual import (SOURCE_STATUS, choose_residual_indices, make_output, read_rows, set_seed)
from src.scripts.run_e018_population_transformer import (EXTERNAL_SOURCES, loss_value, normalized_target, predict, scales)
from src.scripts.run_m3_known_population import decode, expression_metrics
from src.scripts.run_scgpt_roundtrip import NonnegativeLinearDecoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True); p.add_argument("--e95", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--n-cells", type=int, default=512)
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--model-dim", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4); p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--decoder-batch-size", type=int, default=32); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed, args.threads); torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, tokens = np.load(args.cache), np.load(args.tokens)
    external_states = [state_for_source(cache, tokens, source) for source in EXTERNAL_SOURCES]
    external_times = [float(np.unique(cache["stages"][cache["sources"] == source]).item()[1:]) for source in EXTERNAL_SOURCES]
    center_mean, center_scale, dispersion_scale = scales(external_states)
    official_states = [state_for_source(cache, tokens, source) for source in ("E85.h5ad", "E95.h5ad")]
    features, group_ids = token_features(official_states, [8.5, 9.5], 10.5, center_mean, center_scale, dispersion_scale)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = PopulationTransformer(features.shape[1], 512, 50, args.model_dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for target_index in range(2, len(external_states)):
            train_x, train_groups = token_features(external_states[target_index-2:target_index],
                external_times[target_index-2:target_index], external_times[target_index], center_mean, center_scale, dispersion_scale)
            optimizer.zero_grad()
            output = model(torch.tensor(train_x[None], device=device), torch.tensor(train_groups[None], device=device))
            target = normalized_target(external_states[target_index], center_mean, center_scale, dispersion_scale, device)
            loss, _ = loss_value(output, target); loss.backward(); optimizer.step(); epoch_loss += float(loss.detach())
        history.append(epoch_loss / (len(external_states) - 2))
    future = predict(model, features, group_ids, center_mean, center_scale, dispersion_scale, device)

    rng = np.random.default_rng(args.seed)
    counts = rng.multinomial(args.n_cells, future["proportion"])
    generated_groups = np.repeat(np.arange(50), counts).astype(np.int32)
    generated_groups = generated_groups[rng.permutation(args.n_cells)]
    source_mask = cache["sources"] == "E95.h5ad"
    source_latent, source_groups = cache["latent"][source_mask], tokens["labels"][source_mask]
    selected, fallback = choose_residual_indices(generated_groups, source_groups, source_latent, tokens["centers"], args.seed)
    source_mean = official_states[1]["latent_mean"]; source_disp = official_states[1]["latent_dispersion"]
    residual = source_latent[selected] - source_mean[source_groups[selected]]
    scale = future["latent_dispersion"][generated_groups] / np.maximum(source_disp[source_groups[selected]], 1e-4)
    scale = np.clip(scale, 0, 3)
    generated_latent = (future["latent_mean"][generated_groups] + residual * scale).astype(np.float32)

    decoder_data = torch.load(args.decoder, map_location="cpu", weights_only=False)
    decoder = NonnegativeLinearDecoder(decoder_data["input_dim"], decoder_data["output_dim"])
    decoder.load_state_dict(decoder_data["state_dict"])
    raw = decode(decoder, generated_latent, args.decoder_batch_size, device)
    source_decoded = decode(decoder, source_latent[selected], args.decoder_batch_size, device)
    source_expression, genes = read_rows(args.e95, selected)
    corrected = np.maximum(raw + source_expression - source_decoded, 0).astype(np.float32)
    nearest = NearestNeighbors(n_neighbors=1).fit(source_latent).kneighbors(generated_latent, return_distance=False).ravel()
    neighbor, neighbor_genes = read_rows(args.e95, nearest)
    if not np.array_equal(genes, neighbor_genes): raise ValueError("Ordem gênica incompatível.")

    reference_indices = np.sort(rng.choice(len(source_latent), args.n_cells, replace=False))
    reference, _ = read_rows(args.e95, reference_indices)
    variants = {"decoder_raw": raw, "expression_residual_e95": corrected, "nearest_e95": neighbor}
    metrics, contracts = {}, {}
    for name, matrix in variants.items():
        metrics[name] = expression_metrics(matrix, reference, args.seed) | {
            "unique_cells": int(np.unique(matrix, axis=0).shape[0]), "mean_expression": float(matrix.mean()),
            "max_expression": float(matrix.max())}
        output = make_output(matrix, genes, name, args.seed)
        output.uns.update(target_stage="E10.5", experiment="E019", e10_5_target_used=False,
                          diagnostic_reference="E9.5; metrics are distances, not target quality")
        contracts[name] = validate_task1_output(output, genes)
        output.write_h5ad(args.output_dir / f"e10_5_{name}_{args.n_cells}.h5ad", compression="gzip")
    metrics["fallback_cells"] = int(fallback)
    metrics["predicted_proportion_sum"] = float(future["proportion"].sum())
    metrics["predicted_nonempty_groups_sampled"] = int((counts > 0).sum())
    np.savez_compressed(args.output_dir / "predicted_tokens.npz", **future, sampled_counts=counts)
    torch.save({"state_dict": model.state_dict(), "seed": args.seed, "source_status": SOURCE_STATUS}, args.output_dir / "transformer_final.pt")
    config = vars(args) | {"device_used": str(device), "training_sources": EXTERNAL_SOURCES,
        "observed_sources": ["E85.h5ad", "E95.h5ad"], "target_time": 10.5, "target_data_used": False,
        "source_status": SOURCE_STATUS, "training_examples": len(external_states) - 2,
        "normalization": [center_mean, center_scale, dispersion_scale], "latent_residual_scale_clip": [0, 3]}
    for name, value in (("config.json", config), ("metrics.json", metrics), ("contracts.json", contracts),
                        ("training_history.json", history)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, default=str) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
