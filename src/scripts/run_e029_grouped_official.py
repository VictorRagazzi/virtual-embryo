"""E029: shared K-means group transitions using only official E8.5/E9.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from src.approaches.population_transformer.anchored import weighted_population_without_replacement
from src.approaches.population_transformer.nonlinear_pca import (
    ApproximateKernelPCA, GroupTransitionTransformer, decode_variants, decode_zero_weighted_change)
from src.m0_contract import validate_task1_output
from src.scripts.run_e028_official_only import ARCHITECTURES, read_partition, seed_all
from src.scripts.run_m3_known_population import expression_metrics


def group_state(latent: np.ndarray, clusterer: MiniBatchKMeans) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    labels = clusterer.predict(latent); groups, dim = clusterer.n_clusters, latent.shape[1]
    counts = np.bincount(labels, minlength=groups); centers = np.zeros((groups, dim), np.float32)
    dispersion = np.zeros_like(centers)
    for group in np.flatnonzero(counts):
        values = latent[labels == group]; centers[group] = values.mean(0); dispersion[group] = values.std(0)
    state = {"proportion": (counts / counts.sum()).astype(np.float32), "latent_mean": centers,
             "latent_dispersion": dispersion}
    features = np.concatenate([state["proportion"][:, None], centers, dispersion,
                               (counts == 0)[:, None].astype(np.float32)], axis=1)
    return state, features.astype(np.float32), labels


def state_loss(prediction, target, observed):
    loss = (prediction["proportion"] - target["proportion"]).square().mean()
    loss = loss + (prediction["latent_mean"][:, observed] - target["latent_mean"][observed]).square().mean()
    return loss + (prediction["latent_dispersion"][:, observed] - target["latent_dispersion"][observed]).square().mean()


def fit_model(name, architecture, train_features, train_target, validation_features, validation_target,
              validation_observed, args, device):
    torch.manual_seed(args.seed)
    model = GroupTransitionTransformer(args.latent_dim, args.groups, architecture["model_dim"],
                                       architecture["heads"], architecture["layers"], args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ids = torch.arange(args.groups, device=device)[None]
    train_x = torch.tensor(train_features[None], device=device)
    target = {key: torch.tensor(value, device=device) for key, value in train_target.items()}
    train_observed = target["proportion"] > 0
    val_x = torch.tensor(validation_features[None], device=device)
    val_target = {key: torch.tensor(value, device=device) for key, value in validation_target.items()}
    best = float("inf"); bad = 0; history = []; checkpoint = args.output_dir / f"{name}_checkpoint.pt"
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); prediction = model(train_x, ids)
        loss = state_loss(prediction, target, train_observed); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): validation = float(state_loss(model(val_x, ids), val_target, validation_observed))
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_loss": validation})
        if validation < best - 1e-8:
            best = validation; bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation}, checkpoint)
        else:
            bad += 1
            if bad >= args.patience: break
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["state_dict"])
    (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, {"epoch": int(saved["epoch"]), "validation_loss": float(saved["validation_loss"]),
                   "parameters": int(sum(p.numel() for p in model.parameters()))}


def generate(prediction, source_state, source_latent, source_expression, source_labels, encoder, n_cells, seed):
    supported = prediction["proportion"].copy(); supported[np.bincount(source_labels, minlength=len(supported)) == 0] = 0
    supported /= supported.sum()
    selected = weighted_population_without_replacement(supported, source_labels, n_cells, seed)
    groups = source_labels[selected]
    scale = prediction["latent_dispersion"][groups] / np.maximum(source_state["latent_dispersion"][groups], 1e-4)
    latent = prediction["latent_mean"][groups] + (source_latent[selected] - source_state["latent_mean"][groups]) * np.clip(scale, 0, 3)
    matrix = decode_variants(encoder, latent.astype(np.float32), source_latent[selected], source_expression[selected])["decoder_residual"]
    return matrix, groups


def generate_zero_weighted(prediction, source_state, source_latent, source_expression, source_labels,
                           encoder, n_cells, seed, group_support, betas):
    """Generate E029 latents and shrink decoder changes only at anchor zeros."""
    supported = prediction["proportion"].copy()
    supported[np.bincount(source_labels, minlength=len(supported)) == 0] = 0
    supported /= supported.sum()
    selected = weighted_population_without_replacement(supported, source_labels, n_cells, seed)
    groups = source_labels[selected]
    scale = prediction["latent_dispersion"][groups] / np.maximum(source_state["latent_dispersion"][groups], 1e-4)
    latent = prediction["latent_mean"][groups] + (source_latent[selected] - source_state["latent_mean"][groups]) * np.clip(scale, 0, 3)
    matrices = {float(beta): decode_zero_weighted_change(
        encoder, latent.astype(np.float32), source_latent[selected], source_expression[selected],
        groups, group_support, float(beta),
    ) for beta in betas}
    return matrices, groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e85", type=Path, default=Path("data/E85.h5ad")); p.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--train-cells", type=int, default=1024)
    p.add_argument("--validation-cells", type=int, default=256); p.add_argument("--source-pool-cells", type=int, default=2048)
    p.add_argument("--test-cells", type=int, default=512); p.add_argument("--input-genes", type=int, default=512)
    p.add_argument("--landmarks", type=int, default=256); p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--groups", type=int, default=32); p.add_argument("--gamma", type=float, default=.002)
    p.add_argument("--ridge", type=float, default=.01); p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=25); p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    if "_ex" in args.e85.name or "_ex" in args.e95.name: raise ValueError("E029 forbids _ex sources")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    e85, genes = read_partition(args.e85, (args.train_cells, args.validation_cells, args.source_pool_cells), args.seed)
    e95, target_genes = read_partition(args.e95, (args.train_cells, args.validation_cells, args.test_cells), args.seed + 1)
    if not np.array_equal(genes, target_genes): raise ValueError("Gene order mismatch")
    stacked = np.concatenate([e85[0][0], e95[0][0]]); variance = stacked.var(0)
    selected_genes = np.sort(np.argpartition(variance, -args.input_genes)[-args.input_genes:])
    mean = stacked[:, selected_genes].mean(0); scale = stacked[:, selected_genes].std(0) + 1e-4
    normalize = lambda x: ((x[:, selected_genes] - mean) / scale).astype(np.float32)
    encoder = ApproximateKernelPCA(args.landmarks, args.latent_dim, args.gamma, args.ridge, args.seed)
    encoder.fit_embedding(np.concatenate([normalize(e85[0][0]), normalize(e95[0][0])]))
    latent = [[encoder.transform(normalize(part[0])) for part in stage] for stage in (e85, e95)]
    encoder.fit_decoder([latent[0][0], latent[1][0]], [e85[0][0], e95[0][0]])
    clusterer = MiniBatchKMeans(args.groups, random_state=args.seed, n_init=5, batch_size=512).fit(
        np.concatenate([latent[0][0], latent[1][0]]))
    states = [[group_state(latent[stage][part], clusterer) for part in range(3)] for stage in range(2)]
    validation_observed = torch.tensor(states[1][1][0]["proportion"] > 0, device=device)
    predictions = {
        "copy_groups": states[0][2][0],
        "group_delta": {
            "proportion": states[1][0][0]["proportion"].copy(),
            "latent_mean": states[0][2][0]["latent_mean"] + states[1][0][0]["latent_mean"] - states[0][0][0]["latent_mean"],
            "latent_dispersion": np.maximum(states[0][2][0]["latent_dispersion"] + states[1][0][0]["latent_dispersion"] - states[0][0][0]["latent_dispersion"], 0),
        }}
    checkpoints = {}; ids = torch.arange(args.groups, device=device)[None]
    for name, architecture in ARCHITECTURES.items():
        model, checkpoint = fit_model(name, architecture, states[0][0][1], states[1][0][0], states[0][1][1],
                                      states[1][1][0], validation_observed, args, device)
        model.eval()
        with torch.no_grad(): raw = model(torch.tensor(states[0][2][1][None], device=device), ids)
        predictions[name] = {key: value[0].cpu().numpy().astype(np.float32) for key, value in raw.items()}
        checkpoints[name] = checkpoint
    target, target_labels = e95[2]; source, source_labels_text = e85[2]
    metrics = {}; contracts = {}
    for name, prediction in predictions.items():
        matrix, groups = generate(prediction, states[0][2][0], latent[0][2], source, states[0][2][2], encoder,
                                  args.test_cells, args.seed)
        metrics[name] = expression_metrics(matrix, target, args.seed) | {
            "zero_fraction": float((matrix == 0).mean()), "unique_cells": int(np.unique(matrix, axis=0).shape[0])}
        output = ad.AnnData(matrix, obs={"group_id": groups.astype(str)}, var={"gene": genes}); output.var_names = genes
        output.uns.update(experiment="E029", model=name, group_definition="joint_train_E85_E95_KMeans",
                          cell_pairing=False, external_ex_used=False, seed=args.seed)
        contracts[name] = validate_task1_output(output, genes, min_cells=args.test_cells, max_cells=args.test_cells)
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    for name, matrix, labels in (("reference", source[:args.test_cells], source_labels_text[:args.test_cells]),
                                 ("target", target, target_labels)):
        output = ad.AnnData(matrix, obs={"celltype": labels}, var={"gene": genes}); output.var_names = genes
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    result = {"metrics": metrics, "checkpoints": checkpoints, "architectures": ARCHITECTURES,
              "groups": args.groups, "group_definition": "joint KMeans on disjoint E85/E95 training cells",
              "pairing": "same cluster id across stages; no cell pairs", "external_ex_used": False}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "contracts.json").write_text(json.dumps(contracts, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
