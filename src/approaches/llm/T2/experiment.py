"""Reserva amostras independentes e avalia E8.5 → E9.5 com veckit."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .data import prepare_expression
from .temporal import predict


def prepare(data_dir, output, sample_size=2000, seed=42):
    """Reserva células por ID antes do treino; nunca lê matrizes completas."""
    if not 1 <= sample_size <= 2000:
        raise ValueError("A avaliação exige de 1 a 2000 células por amostra.")
    if output.exists():
        raise FileExistsError(output)
    # Nomes explícitos evitam interpretar E675 como 67,5 dias.
    files = [(6.5, "E65_ex.h5ad"), (6.75, "E675_ex.h5ad"), (7.0, "E70_ex.h5ad"),
             (7.25, "E725_ex.h5ad"), (7.5, "E75_ex.h5ad"), (7.75, "E775_ex.h5ad"),
             (8.0, "E80_ex.h5ad"), (8.25, "E825_ex.h5ad"),
             (8.5, "E85.h5ad"), (9.5, "E95.h5ad")]
    manifest = {"seed": seed, "sample_size": sample_size, "stages": [], "evaluation": {}}
    for day, name in files:
        path = (data_dir / name).resolve()
        if not path.exists():
            if day in (8.5, 9.5):
                raise FileNotFoundError(path)
            continue
        manifest["stages"].append(f"{day}={path}")
        if day not in (8.5, 9.5):
            continue
        data = ad.read_h5ad(path, backed="r")
        try:
            if not data.obs_names.is_unique:
                raise ValueError(f"IDs de células repetidos: {path}")
            if data.n_obs < sample_size + 2:
                raise ValueError(f"Poucas células para reserva e treino: {path}")
            indices = np.sort(np.random.default_rng(seed + int(day * 100)).choice(
                data.n_obs, sample_size, replace=False))
            label = "source" if day == 8.5 else "target"
            manifest["evaluation"][label] = {"path": str(path), "time": day,
                                                 "cells": data.obs_names[indices].tolist()}
        finally:
            data.file.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_sample(sample, layer=None):
    """Abre em backed e materializa apenas as células reservadas."""
    cells = sample["cells"]
    if not 1 <= len(cells) <= 2000 or len(set(cells)) != len(cells):
        raise ValueError("Amostra deve conter de 1 a 2000 IDs únicos.")
    data = ad.read_h5ad(sample["path"], backed="r")
    try:
        positions = data.obs_names.get_indexer(cells)
        if (positions < 0).any():
            raise ValueError("IDs reservados ausentes do arquivo original.")
        positions.sort()
        matrix = data.X if layer is None else data.layers[layer]
        subset = ad.AnnData(sparse.csr_matrix(matrix[positions, :], dtype=np.float32),
                            obs=data.obs.iloc[positions].copy(), var=data.var.copy())
        if layer:
            subset.layers[layer] = subset.X.copy()
        return subset
    finally:
        data.file.close()


def evaluate(checkpoint_path, output, batch_size=8, device="cpu"):
    from veckit import score

    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    source = read_sample(manifest["evaluation"]["source"], checkpoint["layer"])
    target = read_sample(manifest["evaluation"]["target"], checkpoint["layer"])
    print(f"Avaliação: {source.n_obs} origens e {target.n_obs} alvos reservados.", flush=True)
    output.mkdir(parents=True)
    source_path = output / "source_raw.h5ad"
    source.write_h5ad(source_path, compression="gzip")
    for label, data in [("reference", source), ("target", target)]:
        positions = data.var_names.get_indexer(checkpoint["genes"])
        if (positions < 0).any():
            raise ValueError("Amostra não contém os genes do checkpoint.")
        _, expression = prepare_expression(data.X, checkpoint["expression_scale"])
        normalized = ad.AnnData(expression[:, positions], obs=data.obs.copy(),
                                var=pd.DataFrame(index=checkpoint["genes"]))
        normalized.uns["expression_scale"] = "log1p_normalized_10000_before_gene_selection"
        normalized.write_h5ad(output / f"{label}.h5ad", compression="gzip")
    del source, target, data, normalized, expression, checkpoint
    predict(argparse.Namespace(checkpoint=checkpoint_path, input=source_path,
                               output=output / "prediction.h5ad", time=8.5, delta_time=1.0,
                               batch_size=batch_size, device=device))
    # T2 é o nome desta abordagem; a tarefa temporal do veckit chama-se T1.
    print("Comparando as distribuições amostradas com veckit (task T1).", flush=True)
    result = score(task="T1", input=output / "prediction.h5ad",
                   target=output / "target.h5ad", reference=output / "reference.h5ad",
                   seed=manifest["seed"], out=output / "metrics.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--data-dir", type=Path, default=Path("data"))
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--sample-size", type=int, default=2000)
    preparation.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.data_dir, args.output, args.sample_size, args.seed)
        print(f"Reserva salva em {args.output}; {len(manifest['stages'])} estágios.")
    else:
        if args.batch_size < 1:
            parser.error("batch-size deve ser positivo")
        evaluate(args.checkpoint, args.output, args.batch_size, args.device)


if __name__ == "__main__":
    main()
