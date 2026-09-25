"""E028: capacity diagnostic using only official E8.5/E9.5 cells."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from src.approaches.population_transformer.nonlinear_pca import (
    ApproximateKernelPCA, SingleStageCellTransformer, decode_variants, distribution_loss)
from src.m0_contract import validate_task1_output
from src.scripts.run_m3_known_population import expression_metrics


ARCHITECTURES = {
    "small": {"model_dim": 64, "heads": 4, "layers": 2},
    "medium": {"model_dim": 128, "heads": 4, "layers": 4},
    "large": {"model_dim": 256, "heads": 8, "layers": 6},
}


def seed_all(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def read_partition(path: Path, sizes: tuple[int, int, int], seed: int):
    data = ad.read_h5ad(path, backed="r"); total = sum(sizes)
    rows = np.random.default_rng(seed).choice(data.n_obs, total, replace=False)
    outputs = []
    offset = 0
    for size in sizes:
        chosen = np.sort(rows[offset:offset + size]); values = data.X[chosen]
        dense = values.toarray() if sp.issparse(values) else np.asarray(values)
        labels = data.obs["celltype"].to_numpy(dtype=str)[chosen]
        outputs.append((dense.astype(np.float32), labels)); offset += size
    genes = data.var_names.to_numpy(dtype=str); data.file.close()
    return outputs, genes


def fit_model(name, architecture, source_train, target_train, source_validation, target_validation,
              directions, args, device):
    torch.manual_seed(args.seed)
    model = SingleStageCellTransformer(args.latent_dim, architecture["model_dim"], architecture["heads"],
                                       architecture["layers"], args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); best = float("inf"); bad = 0; history = []
    checkpoint = args.output_dir / f"{name}_checkpoint.pt"
    validation_source = torch.tensor(source_validation[None], device=device)
    validation_target = torch.tensor(target_validation[None], device=device)
    for epoch in range(args.epochs):
        model.train(); losses = []
        for _ in range(args.steps_per_epoch):
            source_rows = rng.choice(len(source_train), args.batch_cells, replace=False)
            target_rows = rng.choice(len(target_train), args.batch_cells, replace=False)
            source = torch.tensor(source_train[source_rows][None], device=device)
            target = torch.tensor(target_train[target_rows][None], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = distribution_loss(model(source), target, directions); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): validation = float(distribution_loss(model(validation_source), validation_target, directions))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation})
        if validation < best - 1e-7:
            best = validation; bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation}, checkpoint)
        else:
            bad += 1
            if bad >= args.patience: break
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["state_dict"])
    (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, {"epoch": int(saved["epoch"]), "validation_loss": float(saved["validation_loss"]),
                   "parameters": int(sum(value.numel() for value in model.parameters()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e85", type=Path, default=Path("data/E85.h5ad")); p.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--train-cells", type=int, default=1024)
    p.add_argument("--validation-cells", type=int, default=256); p.add_argument("--test-cells", type=int, default=512)
    p.add_argument("--input-genes", type=int, default=512); p.add_argument("--landmarks", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=32); p.add_argument("--gamma", type=float, default=.002)
    p.add_argument("--ridge", type=float, default=.01); p.add_argument("--batch-cells", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=4); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--projections", type=int, default=32)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    if "_ex" in args.e85.name or "_ex" in args.e95.name: raise ValueError("E028 forbids _ex sources")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    sizes = (args.train_cells, args.validation_cells, args.test_cells)
    e85, genes = read_partition(args.e85, sizes, args.seed); e95, target_genes = read_partition(args.e95, sizes, args.seed + 1)
    if not np.array_equal(genes, target_genes): raise ValueError("Gene order mismatch")
    source_train, target_train = e85[0][0], e95[0][0]
    stacked = np.concatenate([source_train, target_train]); variance = stacked.var(0)
    selected = np.sort(np.argpartition(variance, -args.input_genes)[-args.input_genes:])
    mean = stacked[:, selected].mean(0); scale = stacked[:, selected].std(0) + 1e-4
    normalize = lambda value: ((value[:, selected] - mean) / scale).astype(np.float32)
    encoder = ApproximateKernelPCA(args.landmarks, args.latent_dim, args.gamma, args.ridge, args.seed)
    encoder.fit_embedding(np.concatenate([normalize(source_train), normalize(target_train)]))
    latent = [[encoder.transform(normalize(part[0])) for part in stage] for stage in (e85, e95)]
    encoder.fit_decoder([latent[0][0], latent[1][0]], [source_train, target_train])
    directions = torch.nn.functional.normalize(torch.randn(args.projections, args.latent_dim, device=device), dim=1)

    predictions = {"mean_shift": latent[0][2] + latent[1][0].mean(0) - latent[0][0].mean(0)}
    checkpoints = {}
    for name, architecture in ARCHITECTURES.items():
        model, checkpoint = fit_model(name, architecture, latent[0][0], latent[1][0], latent[0][1], latent[1][1],
                                      directions, args, device)
        model.eval()
        with torch.no_grad(): predictions[name] = model(torch.tensor(latent[0][2][None], device=device))[0].cpu().numpy()
        checkpoints[name] = checkpoint

    reference, reference_labels = e85[2]; target, target_labels = e95[2]
    metrics = {}; contracts = {}
    for name, prediction in predictions.items():
        matrix = decode_variants(encoder, prediction, latent[0][2], reference)["decoder_residual"]
        metrics[name] = expression_metrics(matrix, target, args.seed) | {
            "zero_fraction": float((matrix == 0).mean()), "unique_cells": int(np.unique(matrix, axis=0).shape[0])}
        output = ad.AnnData(matrix, var={"gene": genes}); output.var_names = genes
        output.uns.update(experiment="E028", model=name, sources=["E85.h5ad", "E95.h5ad"], external_ex_used=False,
                          interpretation="known_transition_cell_holdout_not_temporal_extrapolation", seed=args.seed)
        contracts[name] = validate_task1_output(output, genes, min_cells=args.test_cells, max_cells=args.test_cells)
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    for name, matrix, labels in (("reference", reference, reference_labels), ("target", target, target_labels)):
        output = ad.AnnData(matrix, obs={"celltype": labels}, var={"gene": genes}); output.var_names = genes
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    result = {"metrics": metrics, "checkpoints": checkpoints, "architectures": ARCHITECTURES,
              "sources": [str(args.e85), str(args.e95)], "external_ex_used": False,
              "interpretation": "capacity on held-out cells from known E8.5->E9.5 transition; not E10.5 predictiveness"}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "contracts.json").write_text(json.dumps(contracts, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
