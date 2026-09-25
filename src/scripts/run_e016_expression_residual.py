"""E016: transfere resíduos de expressão E8.5 para reconstruir E9.5 conhecido."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from src.m0_contract import validate_task1_output
from src.scripts.run_m3_known_population import decode, expression_metrics
from src.scripts.run_scgpt_roundtrip import NonnegativeLinearDecoder


SOURCE_STATUS = "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def choose_residual_indices(
    target_groups: np.ndarray,
    source_groups: np.ndarray,
    source_latent: np.ndarray,
    centers: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Amostra fonte do mesmo grupo; vazio usa grupo não vazio de centro mais próximo."""
    rng = np.random.default_rng(seed)
    pools = {int(g): np.flatnonzero(source_groups == g) for g in np.unique(source_groups)}
    chosen = np.empty(len(target_groups), dtype=np.int64)
    fallback = 0
    nonempty = np.asarray(sorted(pools), dtype=np.int64)
    for group in np.unique(target_groups):
        output = np.flatnonzero(target_groups == group)
        source_group = int(group)
        if source_group not in pools:
            distances = np.linalg.norm(centers[nonempty] - centers[source_group], axis=1)
            source_group = int(nonempty[np.argmin(distances)])
            fallback += len(output)
        chosen[output] = rng.choice(pools[source_group], len(output), replace=True)
    return chosen, fallback


def read_rows(path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r")
    unique, inverse = np.unique(indices, return_inverse=True)
    matrix = data.X[unique]
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    genes = data.var_names.to_numpy(dtype=str)
    data.file.close()
    return dense[inverse].astype(np.float32), genes


def make_output(matrix: np.ndarray, genes: np.ndarray, method: str, seed: int) -> ad.AnnData:
    output = ad.AnnData(X=np.maximum(matrix, 0).astype(np.float32), var={"gene": genes})
    output.var_names = genes
    output.obs["generator"] = method
    output.uns.update(seed=seed, source_status=SOURCE_STATUS, experiment="E016")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--e85", type=Path, required=True)
    parser.add_argument("--e95", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--decoder-batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, tokens = np.load(args.cache), np.load(args.tokens)
    latent, stages, sources, labels = cache["latent"], cache["stages"], cache["sources"], tokens["labels"]
    decoder_data = torch.load(args.decoder, map_location="cpu", weights_only=False)
    if decoder_data["selected_genes"] != cache["selected_genes"].tolist():
        raise ValueError("Tokenização do cache e decoder incompatíveis.")
    decoder = NonnegativeLinearDecoder(decoder_data["input_dim"], decoder_data["output_dim"])
    decoder.load_state_dict(decoder_data["state_dict"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    rng = np.random.default_rng(args.seed)
    source85 = np.flatnonzero(sources == args.e85.name)
    source95 = np.flatnonzero(sources == args.e95.name)
    if len(source85) < args.n_cells or len(source95) < args.n_cells:
        raise ValueError("Amostra maior que o estágio disponível.")
    local85 = np.sort(rng.choice(len(source85), args.n_cells, replace=False))
    local95 = np.sort(rng.choice(len(source95), args.n_cells, replace=False))
    index85, index95 = source85[local85], source95[local95]
    x85, genes85 = read_rows(args.e85, local85)
    x95, genes95 = read_rows(args.e95, local95)
    if not np.array_equal(genes85, genes95):
        raise ValueError("Genes E8.5/E9.5 incompatíveis.")

    z85, z95 = latent[index85], latent[index95]
    group85, group95 = labels[index85], labels[index95]
    decoded85 = decode(decoder, z85, args.decoder_batch_size, device)
    decoded95 = decode(decoder, z95, args.decoder_batch_size, device)
    residual85 = x85 - decoded85
    residual_indices, fallback = choose_residual_indices(group95, group85, z85, tokens["centers"], args.seed)
    corrected = np.maximum(decoded95 + residual85[residual_indices], 0).astype(np.float32)

    centroid_latent = np.stack([z95[group95 == group].mean(0) if np.any(group95 == group) else tokens["centers"][group]
                                for group in group95]).astype(np.float32)
    centroid = decode(decoder, centroid_latent, args.decoder_batch_size, device)
    nearest = NearestNeighbors(n_neighbors=1).fit(z85).kneighbors(z95, return_distance=False).ravel()
    neighbor = x85[nearest]
    variants = {"decoder_raw": decoded95, "expression_residual_e85": corrected,
                "centroid": centroid, "nearest_e85": neighbor}
    metrics: dict[str, object] = {name: expression_metrics(value, x95, args.seed) | {
        "unique_cells": int(np.unique(value, axis=0).shape[0])} for name, value in variants.items()}
    metrics.update(fallback_cells=int(fallback), n_cells=args.n_cells)
    contracts = {}
    for name, value in variants.items():
        output = make_output(value, genes95, name, args.seed)
        contracts[name] = validate_task1_output(output, genes95)
        output.write_h5ad(args.output_dir / f"e95_{name}_{args.n_cells}.h5ad", compression="gzip")
    config = vars(args) | {"device_used": str(device), "source_status": SOURCE_STATUS,
                           "residual_source": "D001/E8.5 only", "target_role": "D002 known pseudo-holdout"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
