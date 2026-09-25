"""E027: approximate kernel PCA with grouped-summary versus raw-cell Transformers."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans

from src.approaches.population_transformer.nonlinear_pca import (
    ApproximateKernelPCA, CellSetTransformer, GroupSummaryTransformer,
    decode_variants, distribution_loss)
from src.m0_contract import validate_task1_output
from src.scripts.run_e021_anchored import EXTERNAL
from src.scripts.run_m3_known_population import expression_metrics


def seed_all(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def read_sample(path: Path, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r")
    rows = np.sort(np.random.default_rng(seed).choice(data.n_obs, min(n, data.n_obs), replace=False))
    values = data.X[rows]
    dense = values.toarray() if sp.issparse(values) else np.asarray(values)
    genes = data.var_names.to_numpy(dtype=str)
    labels = data.obs["celltype"].to_numpy(dtype=str)[rows] if "celltype" in data.obs else np.repeat("NA", len(rows))
    data.file.close()
    return dense.astype(np.float32), genes, labels


def group_summary(latent: np.ndarray, clusterer: MiniBatchKMeans) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = clusterer.predict(latent); groups = clusterer.n_clusters; dim = latent.shape[1]
    result = np.zeros((groups, 2 * dim + 1), dtype=np.float32)
    centers = np.zeros((groups, dim), dtype=np.float32)
    counts = np.bincount(labels, minlength=groups)
    for group in range(groups):
        values = latent[labels == group]
        center = values.mean(0) if len(values) else clusterer.cluster_centers_[group]
        spread = values.std(0) if len(values) else np.zeros(dim, dtype=np.float32)
        centers[group] = center
        result[group] = np.concatenate([[counts[group] / len(labels)], center, np.asarray(spread).reshape(-1)])
    return result, centers, labels


def train_raw(latents: list[np.ndarray], args, device, directions):
    model = CellSetTransformer(args.latent_dim, args.model_dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); examples = []
    for target in range(2, len(latents) - 1):
        picks = [rng.choice(len(latents[i]), args.train_cells, replace=len(latents[i]) < args.train_cells)
                 for i in (target-2, target-1, target)]
        examples.append(tuple(torch.tensor(latents[i][pick], device=device)[None] for i, pick in zip((target-2,target-1,target), picks)))
    return train_loop(model, optimizer, examples, directions, args, "raw")


def train_grouped(summaries, centers, args, device, directions):
    model = GroupSummaryTransformer(args.latent_dim, args.model_dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    examples = [(torch.tensor(summaries[t-2], device=device)[None],
                 torch.tensor(summaries[t-1], device=device)[None],
                 torch.tensor(centers[t], device=device)[None]) for t in range(2, len(summaries)-1)]
    return train_loop(model, optimizer, examples, directions, args, "grouped")


def train_loop(model, optimizer, examples, directions, args, name):
    best = float("inf"); bad = 0; history = []; checkpoint = args.output_dir / f"{name}_checkpoint.pt"
    for epoch in range(args.epochs):
        model.train(); values = []
        for first, second, target in examples:
            optimizer.zero_grad(set_to_none=True); prediction = model(first, second)
            loss = distribution_loss(prediction, target, directions)
            loss.backward(); optimizer.step(); values.append(float(loss.detach()))
        value = float(np.mean(values)); history.append({"epoch": epoch, "loss": value})
        if value < best - 1e-7:
            best = value; bad = 0; torch.save({"state_dict": model.state_dict(), "epoch": epoch, "loss": value}, checkpoint)
        else:
            bad += 1
            if bad >= args.patience: break
    saved = torch.load(checkpoint, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--evaluation-reference", type=Path,
                   default=Path("outputs/population_transformer/E021/fixed_eval_seed42/reference.h5ad"))
    p.add_argument("--evaluation-target", type=Path,
                   default=Path("outputs/population_transformer/E021/fixed_eval_seed42/target.h5ad"))
    p.add_argument("--fit-cells", type=int, default=256); p.add_argument("--input-genes", type=int, default=512)
    p.add_argument("--landmarks", type=int, default=256); p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--groups", type=int, default=32); p.add_argument("--train-cells", type=int, default=128)
    p.add_argument("--n-cells", type=int, default=512); p.add_argument("--gamma", type=float, default=0.002)
    p.add_argument("--ridge", type=float, default=0.01); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.05); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--projections", type=int, default=32); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); seed_all(args.seed, args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    blocks = []; genes = None
    for index, name in enumerate(EXTERNAL[:-1]):
        block, current_genes, _ = read_sample(args.data_dir / name, args.fit_cells, args.seed + index)
        if genes is not None and not np.array_equal(genes, current_genes): raise ValueError("Gene order mismatch")
        blocks.append(block); genes = current_genes
    stacked = np.concatenate(blocks); variance = stacked.var(0)
    selected = np.sort(np.argpartition(variance, -args.input_genes)[-args.input_genes:])
    selected_data = stacked[:, selected]; mean = selected_data.mean(0); scale = selected_data.std(0) + 1e-4
    normalized = [((block[:, selected] - mean) / scale).astype(np.float32) for block in blocks]
    nonlinear = ApproximateKernelPCA(args.landmarks, args.latent_dim, args.gamma, args.ridge, args.seed)
    nonlinear.fit_embedding(np.concatenate(normalized)); latents = [nonlinear.transform(value) for value in normalized]
    nonlinear.fit_decoder(latents, blocks)

    reference, reference_genes, reference_labels = read_sample(args.evaluation_reference, args.n_cells, args.seed + 100)
    target, target_genes, target_labels = read_sample(args.evaluation_target, args.n_cells, args.seed + 101)
    if not np.array_equal(genes, reference_genes) or not np.array_equal(genes, target_genes): raise ValueError("Gene order mismatch")
    reference_latent = nonlinear.transform(((reference[:, selected] - mean) / scale).astype(np.float32))
    target_latent = nonlinear.transform(((target[:, selected] - mean) / scale).astype(np.float32))

    clusterer = MiniBatchKMeans(args.groups, random_state=args.seed, n_init=5, batch_size=512).fit(np.concatenate(latents))
    summary_data = [group_summary(value, clusterer) for value in latents]
    summaries = [value[0] for value in summary_data]; centers = [value[1] for value in summary_data]
    directions = torch.nn.functional.normalize(torch.randn(args.projections, args.latent_dim, device=device), dim=1)
    raw_model, raw_saved = train_raw(latents, args, device, directions)
    grouped_model, grouped_saved = train_grouped(summaries, centers, args, device, directions)

    raw_model.eval(); grouped_model.eval(); rng = np.random.default_rng(args.seed)
    previous_two = latents[-2][rng.choice(len(latents[-2]), args.n_cells, replace=True)]
    with torch.no_grad():
        raw_prediction = raw_model(torch.tensor(previous_two, device=device)[None],
                                   torch.tensor(reference_latent, device=device)[None])[0].cpu().numpy()
    ref_summary, ref_centers, ref_labels = group_summary(reference_latent, clusterer)
    with torch.no_grad():
        predicted_centers = grouped_model(torch.tensor(summaries[-2], device=device)[None],
                                          torch.tensor(ref_summary, device=device)[None])[0].cpu().numpy()
    grouped_prediction = reference_latent + predicted_centers[ref_labels] - ref_centers[ref_labels]

    all_metrics = {}; contracts = {}
    for approach, predicted in {"grouped": grouped_prediction, "raw_cells": raw_prediction}.items():
        for decoder, matrix in decode_variants(nonlinear, predicted, reference_latent, reference).items():
            name = f"{approach}__{decoder}"; metrics = expression_metrics(matrix, target, args.seed)
            metrics.update(zero_fraction=float((matrix == 0).mean()), unique_cells=int(np.unique(matrix, axis=0).shape[0]))
            all_metrics[name] = metrics
            output = ad.AnnData(matrix.astype(np.float32), var={"gene": genes}); output.var_names = genes
            output.uns.update(experiment="E027", approach=approach, decoder=decoder, seed=args.seed,
                              pseudo_target=EXTERNAL[-1], paired_cells=False)
            contracts[name] = validate_task1_output(output, genes, min_cells=args.n_cells, max_cells=args.n_cells)
            output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    for name, matrix, labels in (("reference", reference, reference_labels), ("target", target, target_labels)):
        output = ad.AnnData(matrix, obs={"celltype": labels}, var={"gene": genes}); output.var_names = genes
        output.write_h5ad(args.output_dir / f"{name}.h5ad", compression="gzip")
    result = {"metrics": all_metrics,
              "raw_checkpoint": {"epoch": int(raw_saved["epoch"]), "loss": float(raw_saved["loss"])},
              "grouped_checkpoint": {"epoch": int(grouped_saved["epoch"]), "loss": float(grouped_saved["loss"])},
              "embedding": {"method": "RBF_Nystroem_plus_PCA", "selected_genes": args.input_genes,
                            "landmarks": args.landmarks, "latent_dim": args.latent_dim, "gamma": args.gamma},
              "device": str(device), "no_cell_pairing": True}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "contracts.json").write_text(json.dumps(contracts, indent=2, default=str) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    np.savez_compressed(args.output_dir / "embedding.npz", selected_genes=selected, mean=mean, scale=scale,
                        reference_latent=reference_latent, target_latent=target_latent)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
