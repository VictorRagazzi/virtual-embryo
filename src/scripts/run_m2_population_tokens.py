"""M2 exploratório: cache CLS scGPT, K-means conjunto e tokens populacionais."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from src.approaches.llm.T1.fine_tuning.config import Config
from src.approaches.llm.T1.fine_tuning.scgpt_model import ScGPTRegressor, load_pretrained_weights
from src.scripts.run_scgpt_roundtrip import build_sequences, extract_embeddings, token_id


SOURCE_STATUS = "PERMITIDO_POR_DECISAO_DO_RESPONSAVEL_RISCO_RESIDUAL_DOCUMENTADO"


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def stage_from_path(path: Path) -> str:
    name = path.stem.removesuffix("_ex")
    digits = name.removeprefix("E")
    return f"E{digits[0]}.{digits[1:]}" if len(digits) > 1 else f"E{digits}"


def validate_sources(paths: list[Path]) -> tuple[list[str], str]:
    genes: list[str] | None = None
    for path in paths:
        data = ad.read_h5ad(path, backed="r")
        current = data.var_names.astype(str).tolist()
        if genes is None:
            genes = current
        elif current != genes:
            raise ValueError(f"Ordem de genes incompatível: {path}")
        if data.X.dtype != np.float32:
            raise ValueError(f"X precisa ser float32: {path}")
        data.file.close()
    assert genes is not None
    return genes, hashlib.sha256("\n".join(genes).encode()).hexdigest()


def select_genes(paths: list[Path], genes: list[str], vocab: dict[str, int], count: int, block: int) -> list[str]:
    total = 0
    sums = np.zeros(len(genes), dtype=np.float64)
    squares = np.zeros(len(genes), dtype=np.float64)
    for path in paths:
        data = ad.read_h5ad(path, backed="r")
        source_matrix = data.X[:]
        for start in range(0, data.n_obs, block):
            x = source_matrix[start : start + block]
            sums += np.asarray(x.sum(axis=0)).ravel()
            squares += np.asarray(x.multiply(x).sum(axis=0)).ravel() if hasattr(x, "multiply") else np.square(np.asarray(x)).sum(axis=0)
            total += x.shape[0]
        data.file.close()
    variance = np.maximum(squares / total - np.square(sums / total), 0)
    candidates = [i for i, gene in enumerate(genes) if gene in vocab or gene.upper() in vocab]
    chosen = sorted(candidates, key=lambda i: (-variance[i], genes[i]))[:count]
    if len(chosen) != count:
        raise ValueError(f"Somente {len(chosen)} genes mapeiam no vocabulário.")
    return [genes[i] for i in chosen]


def extract_all(paths: list[Path], genes: list[str], selected: list[str], vocab: dict[str, int], model: ScGPTRegressor,
                batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.to(device).eval()
    latents, stages, sources, programs = [], [], [], []
    positions = [genes.index(gene) for gene in selected]
    for path in paths:
        data = ad.read_h5ad(path, backed="r")
        stage = str(data.obs["stage"].iloc[0]) if "stage" in data.obs else stage_from_path(path)
        source_matrix = data.X[:]
        for start in range(0, data.n_obs, batch_size):
            matrix = source_matrix[start : start + batch_size]
            ids, values, mask = build_sequences(matrix, genes, selected, vocab, Config.N_BINS)
            with torch.no_grad():
                states = model.encode(torch.from_numpy(ids).to(device), torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device))
            latents.append(states[:, 0].cpu().numpy().astype(np.float32))
            subset = matrix[:, positions]
            programs.append((subset.toarray() if hasattr(subset, "toarray") else np.asarray(subset)).astype(np.float32))
            stages.extend([stage] * matrix.shape[0])
            sources.extend([path.name] * matrix.shape[0])
        data.file.close()
    return np.concatenate(latents), np.asarray(stages), np.asarray(sources), np.concatenate(programs)


def matched_proportion_l1(reference: KMeans, other: KMeans, ref_labels: np.ndarray, labels: np.ndarray) -> float:
    distances = np.linalg.norm(reference.cluster_centers_[:, None] - other.cluster_centers_[None, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    p = np.bincount(ref_labels, minlength=len(rows)) / len(ref_labels)
    q_raw = np.bincount(labels, minlength=len(rows)) / len(labels)
    q = np.empty_like(q_raw, dtype=float)
    q[rows] = q_raw[cols]
    return float(np.abs(p - q).sum())


def summarize(latent: np.ndarray, programs: np.ndarray, stages: np.ndarray, labels: np.ndarray, k: int,
              selected: list[str]) -> dict[str, np.ndarray]:
    unique_stages = np.asarray(sorted(set(stages), key=lambda x: float(x[1:])))
    shape = (len(unique_stages), k)
    counts = np.zeros(shape, dtype=np.int64)
    means = np.zeros(shape + (latent.shape[1],), dtype=np.float32)
    dispersion = np.zeros_like(means)
    scores = np.zeros(shape + (programs.shape[1],), dtype=np.float32)
    top_n = min(10, len(selected))
    top_genes = np.full(shape + (top_n,), "", dtype=f"U{max(map(len, selected))}")
    for si, stage in enumerate(unique_stages):
        stage_mask = stages == stage
        for group in range(k):
            mask = stage_mask & (labels == group)
            counts[si, group] = mask.sum()
            if counts[si, group]:
                means[si, group] = latent[mask].mean(axis=0)
                dispersion[si, group] = latent[mask].std(axis=0)
                scores[si, group] = programs[mask].mean(axis=0)
                top = np.argsort(-scores[si, group], kind="stable")[:top_n]
                top_genes[si, group] = np.asarray(selected)[top]
    proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    stage_index = {stage: index for index, stage in enumerate(unique_stages)}
    residuals = np.stack([latent[i] - means[stage_index[stage], labels[i]] for i, stage in enumerate(stages)]).astype(np.float32)
    return {"stage_names": unique_stages, "cell_count": counts, "proportion": proportions.astype(np.float32),
            "latent_mean": means, "latent_dispersion": dispersion, "gene_program_scores": scores,
            "program_gene_names": np.asarray(selected), "top_genes": top_genes, "empty_mask": counts == 0,
            "latent_residual": residuals}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Config.CHECKPOINT_PATH)
    parser.add_argument("--vocab", type=Path, default=Config.VOCAB_PATH)
    parser.add_argument("--k", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--input-genes", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--scan-block-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--selected-genes-from", type=Path, help="Checkpoint de decoder cujo conjunto de genes deve ser reutilizado.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seeds[0], args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes, gene_hash = validate_sources(args.inputs)
    vocab = json.loads(args.vocab.read_text())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    cache_path = args.output_dir / "embeddings.npz"
    if args.reuse_embeddings:
        cache = np.load(cache_path)
        latent, stages, sources = cache["latent"], cache["stages"], cache["sources"]
        programs, selected = cache["program_expression"], cache["selected_genes"].tolist()
    else:
        if args.selected_genes_from:
            decoder_metadata = torch.load(args.selected_genes_from, map_location="cpu", weights_only=False)
            selected = list(decoder_metadata["selected_genes"])
            if len(selected) != args.input_genes:
                raise ValueError("O checkpoint não contém o número solicitado de genes.")
        else:
            selected = select_genes(args.inputs, genes, vocab, args.input_genes, args.scan_block_size)
        model = ScGPTRegressor(len(vocab), Config.D_MODEL, Config.N_HEAD, Config.D_HID, Config.N_LAYERS, Config.DROPOUT)
        load_pretrained_weights(model, args.checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad = False
        latent, stages, sources, programs = extract_all(args.inputs, genes, selected, vocab, model, args.embedding_batch_size, device)
        np.savez_compressed(cache_path, latent=latent, stages=stages, sources=sources,
                            program_expression=programs, selected_genes=np.asarray(selected))
    metrics: dict[str, object] = {"n_cells": len(latent), "latent_dim": latent.shape[1], "device": str(device), "k": {}}
    for k in args.k:
        models, labels = [], []
        for seed in args.seeds:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048,
                                 max_iter=100, reassignment_ratio=0.01).fit(latent)
            models.append(km); labels.append(km.labels_.astype(np.int32))
        summary = summarize(latent, programs, stages, labels[0], k, selected)
        np.savez_compressed(args.output_dir / f"tokens_k{k}.npz", labels=labels[0], centers=models[0].cluster_centers_.astype(np.float32), **summary)
        aris = [adjusted_rand_score(labels[0], label) for label in labels[1:]]
        prop_l1 = [matched_proportion_l1(models[0], model_i, labels[0], label) for model_i, label in zip(models[1:], labels[1:])]
        counts = summary["cell_count"]
        metrics["k"][str(k)] = {"inertia_per_cell": float(models[0].inertia_ / len(latent)),
            "ari_vs_seed_42": aris, "ari_mean": float(np.mean(aris)), "matched_global_proportion_l1": prop_l1,
            "empty_stage_groups": int((counts == 0).sum()), "small_stage_groups_lt10": int(((counts > 0) & (counts < 10)).sum()),
            "min_nonempty_group": int(counts[counts > 0].min()), "max_group": int(counts.max()),
            "mean_latent_dispersion": float(summary["latent_dispersion"][counts > 0].mean()),
            "proportion_sum_max_error": float(np.abs(summary["proportion"].sum(1) - 1).max())}
    config = {"seed": args.seeds[0], "seeds": args.seeds, "inputs": list(map(str, args.inputs)), "k": args.k,
              "input_genes": args.input_genes, "embedding_batch_size": args.embedding_batch_size,
              "selected_genes_from": str(args.selected_genes_from) if args.selected_genes_from else None,
              "clustering": "MiniBatchKMeans(n_init=3,batch_size=2048,max_iter=100)",
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "gene_order_sha256": gene_hash,
              "selected_gene_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(), "source_status": SOURCE_STATUS}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
