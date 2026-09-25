"""Config-driven E020 pipeline and cheap end-to-end smoke test."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

from src.approaches.population_transformer.objectives import RFFMMD, de_ranking_loss, differentiable_population
from src.approaches.population_transformer.dataset_v2 import build_tasks
from src.approaches.population_transformer.pipeline_v2 import metric_errors, train_run
from src.approaches.population_transformer.objectives import selection_score
from src.approaches.population_transformer.temporal import PopulationTransformer, copy_last, state_for_source, token_features
from src.scripts.run_e018_population_transformer import normalized_target, scales
from src.scripts.run_e016_expression_residual import choose_residual_indices, read_rows
from src.scripts.run_m3_known_population import decode
from src.scripts.run_scgpt_roundtrip import NonnegativeLinearDecoder
from src.m0_contract import validate_task1_output


def load_config(path: Path, smoke: bool) -> dict:
    config = yaml.safe_load(path.read_text())
    if smoke:
        config = config | {"mode": "smoke"}
        config["training"] = config["training"] | {"epochs": 2, "patience": 1}
        config["loss"] = config["loss"] | {"ranking_genes": 32, "ranking_block_size": 8,
                                             "rff_features": 16, "top_k": 8}
        config["generation"] = config["generation"] | {"n_cells": 24}
    return config


def validate_inputs(config: dict, *, smoke: bool) -> None:
    if smoke:
        return
    missing = [str(path) for path in map(Path, config["paths"]["required"]) if not path.exists()]
    if missing:
        raise FileNotFoundError("Required data/weights are absent; copy or regenerate them: " + ", ".join(missing))


def run_smoke(config: dict, output: Path) -> dict:
    """Exercise gradients, checkpointing and all three export variants on synthetic data."""
    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    cells, genes, latent, groups = config["generation"]["n_cells"], config["loss"]["ranking_genes"], 8, 5
    centers = torch.nn.Parameter(torch.randn(groups, latent) * .1)
    raw_dispersion = torch.nn.Parameter(torch.zeros(groups, latent))
    decoder_weight = torch.randn(genes, latent) * .1
    decoder_bias = torch.zeros(genes)
    group_ids = torch.arange(cells) % groups
    residuals = torch.randn(cells, latent) * .2
    reference = torch.rand(cells, genes)
    target = torch.clamp(reference + torch.linspace(-.15, .15, genes), min=0)
    mmd = RFFMMD(genes, config["loss"]["rff_features"], config["loss"]["rff_sigma"], seed)
    optimizer = torch.optim.AdamW([centers, raw_dispersion], lr=1e-2)
    history = []
    for epoch in range(config["training"]["epochs"]):
        optimizer.zero_grad()
        predicted = differentiable_population(centers, torch.nn.functional.softplus(raw_dispersion), group_ids,
                                                residuals, decoder_weight, decoder_bias)
        rank = de_ranking_loss(predicted, target, reference, temperature=config["loss"]["ranking_temperature"],
                               top_k=config["loss"]["top_k"], block_size=config["loss"]["ranking_block_size"])
        distribution = mmd(predicted, target)
        loss = config["loss"]["ranking_weight"] * rank + config["loss"]["mmd_weight"] * distribution
        loss.backward(); optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "de_ranking": float(rank.detach()),
                        "rff_mmd": float(distribution.detach())})
    torch.save({"centers": centers.detach(), "dispersion": torch.nn.functional.softplus(raw_dispersion.detach()),
                "optimizer": optimizer.state_dict(), "epoch": len(history), "seed": seed}, output / "checkpoint.pt")
    raw = predicted.detach().numpy().astype(np.float32)
    residual = np.maximum(raw + reference.numpy() - raw.mean(0), 0).astype(np.float32)
    nearest = reference.numpy().astype(np.float32)
    names = np.asarray([f"gene_{index}" for index in range(genes)], dtype=str)
    contracts = {}
    for variant, matrix in {"decoder_raw": raw, "expression_residual_e95": residual, "nearest_e95": nearest}.items():
        item = ad.AnnData(matrix)
        item.var_names = names
        item.obs_names = [f"synthetic_{index}" for index in range(cells)]
        item.uns.update(experiment="E020_SMOKE", seed=seed, variant=variant, sources=["synthetic_smoke"])
        contracts[variant] = validate_task1_output(item, names, min_cells=cells, max_cells=cells)
        item.write_h5ad(output / f"{variant}.h5ad")
    metrics = {"smoke_only": True, "final_loss": history[-1], "contracts": contracts,
               "finite_gradients": all(p.grad is not None and torch.isfinite(p.grad).all() for p in (centers, raw_dispersion))}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"artifacts": sorted(p.name for p in output.iterdir()),
                                                        "resume": str(output / "checkpoint.pt")}, indent=2) + "\n")
    return metrics


def run_search(config: dict, output: Path, resume: Path | None) -> dict:
    paths = config["paths"]
    artifacts = [Path(paths[name]) for name in ("embeddings", "tokens", "decoder")]
    if any(not path.exists() for path in artifacts):
        raise FileNotFoundError("Caches persistentes ausentes. Execute os comandos de preparação documentados: " +
                                ", ".join(str(path) for path in artifacts if not path.exists()))
    cache, tokens = np.load(artifacts[0]), np.load(artifacts[1])
    decoder_data = torch.load(artifacts[2], map_location="cpu", weights_only=False)
    external = config["data"]["external_sources"]
    data_dir = Path(config["paths"].get("data_dir", "data"))
    source_paths = {name: data_dir / name for name in external}
    source_paths.update({"E85.h5ad": data_dir / "E85.h5ad", "E95.h5ad": data_dir / "E95.h5ad"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples, validation_item, metadata = build_tasks(cache=cache, tokens=tokens, decoder_data=decoder_data,
        source_paths=source_paths, sequence=external, holdout_source=config["data"]["temporal_holdout"],
        official_sources=("E85.h5ad", "E95.h5ad"), official_holdout=config["data"]["official_holdout_cells"],
        ranking_genes=config["loss"]["ranking_genes"], loss_cells=config["training"]["loss_cells"],
        seed=config["seed"], device=device)
    norm = metadata["normalization"]
    def execute(run_dir, architecture, current_loss, seed, score_fn):
        fallbacks = [current_loss["ranking_block_size"]] + [int(v) for v in config["training"].get("oom_block_fallbacks", [])]
        events = []
        for block in dict.fromkeys(fallbacks):
            attempted = current_loss.copy() | {"ranking_block_size": block}
            try:
                result = train_run(examples=examples, validation={"score_fn": score_fn}, architecture=architecture,
                    loss_config=attempted, training=config["training"], output=run_dir, seed=seed,
                    resume=(run_dir / "last.pt") if config["training"]["resume"] else None, device=device)
                if events: (run_dir / "oom_fallbacks.json").write_text(json.dumps(events, indent=2) + "\n")
                return result
            except torch.OutOfMemoryError as error:
                events.append({"failed_block_size": block, "error": str(error)})
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise RuntimeError(f"OOM persisted for declared block fallbacks: {events}")
    def score_factory(loss_cfg):
        rff_eval = RFFMMD(loss_cfg["ranking_genes"], loss_cfg["rff_features"], loss_cfg["rff_sigma"], config["seed"]).to(device)
        previous = validation_item["previous_state"]
        base_expression = differentiable_population(torch.tensor(previous["latent_mean"], device=device),
            torch.tensor(previous["latent_dispersion"], device=device), validation_item["draw_groups"].to(device),
            validation_item["residuals"].to(device), validation_item["decoder_weight"].to(device),
            validation_item["decoder_bias"].to(device)).detach().cpu().numpy()
        base_errors = metric_errors(copy_last(previous), validation_item["target"], base_expression,
            validation_item["target_expression"].numpy(), validation_item["reference_expression"].numpy(), rff_eval.cpu(),
            loss_cfg["ranking_temperature"], loss_cfg["top_k"], loss_cfg["ranking_block_size"])
        rff_eval.to(device)
        def score(model, _):
            model.eval()
            with torch.no_grad():
                raw = model(torch.tensor(validation_item["features"][None], device=device),
                            torch.tensor(validation_item["groups"][None], device=device))
                centers = raw["latent_mean"][0] * norm[1] + norm[0]; dispersion = raw["latent_dispersion"][0] * norm[2]
                expression = differentiable_population(centers, dispersion, validation_item["draw_groups"].to(device),
                    validation_item["residuals"].to(device), validation_item["decoder_weight"].to(device),
                    validation_item["decoder_bias"].to(device))
            prediction = {"proportion": raw["proportion"][0].cpu().numpy(), "latent_mean": centers.cpu().numpy(),
                          "latent_dispersion": dispersion.cpu().numpy()}
            errors = metric_errors(prediction, validation_item["target"], expression.cpu().numpy(),
                validation_item["target_expression"].numpy(), validation_item["reference_expression"].numpy(),
                rff_eval.cpu(), loss_cfg["ranking_temperature"], loss_cfg["top_k"], loss_cfg["ranking_block_size"])
            rff_eval.to(device); return selection_score(errors, base_errors)["score"]
        return score, base_errors
    results = []
    base_loss = config["loss"].copy(); base_loss["ranking_weight"] = base_loss["mmd_weight"] = 0.
    for index, arch in enumerate(config["model"]["architectures"]):
        architecture = arch | {"dropout": config["model"]["dropout"]}; score_fn, baseline = score_factory(base_loss)
        run_dir = output / "phase1" / f"a{index}"
        result = execute(run_dir, architecture, base_loss, config["seed"], score_fn)
        results.append({"phase": 1, "architecture": architecture, "loss": "structural", "score": result.best_score,
                        "checkpoint": str(result.checkpoint), "baseline_errors": baseline})
    best_arch = max(results, key=lambda item: item["score"])["architecture"]
    variants = [("structural", 0., 0.), ("ranking", config["loss"]["ranking_weight"], 0.),
                ("mmd", 0., config["loss"]["mmd_weight"]),
                ("ranking_mmd", config["loss"]["ranking_weight"], config["loss"]["mmd_weight"])]
    for name, rank_weight, mmd_weight in variants:
        current = config["loss"].copy() | {"ranking_weight": rank_weight, "mmd_weight": mmd_weight}
        score_fn, baseline = score_factory(current); run_dir = output / "phase2" / name
        result = execute(run_dir, best_arch, current, config["seed"], score_fn)
        results.append({"phase": 2, "architecture": best_arch, "loss": name, "score": result.best_score,
                        "checkpoint": str(result.checkpoint), "baseline_errors": baseline})
    for index, multiplier in enumerate((.5, 2.)):
        current = config["loss"].copy() | {"ranking_weight": config["loss"]["ranking_weight"] * multiplier,
                                           "mmd_weight": config["loss"]["mmd_weight"] * multiplier}
        score_fn, baseline = score_factory(current); run_dir = output / "phase2" / f"weights_{multiplier:g}"
        result = execute(run_dir, best_arch, current, config["seed"], score_fn)
        results.append({"phase": 2, "architecture": best_arch, "loss": f"ranking_mmd_x{multiplier:g}",
                        "score": result.best_score, "checkpoint": str(result.checkpoint), "baseline_errors": baseline})
    best = max((item for item in results if item["phase"] == 2), key=lambda item: item["score"])
    best_checkpoint = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    stability_loss = best_checkpoint["loss"]; score_fn, baseline = score_factory(stability_loss)
    stability_dir = output / "phase2" / "stability_seed43"
    stability = execute(stability_dir, best_arch, stability_loss, config["training"]["seeds"][1], score_fn)
    results.append({"phase": 2, "architecture": best_arch, "loss": best["loss"] + "_seed43", "score": stability.best_score,
                    "checkpoint": str(stability.checkpoint), "baseline_errors": baseline, "stability_repeat": True})
    output.mkdir(parents=True, exist_ok=True)
    (output / "search_results.json").write_text(json.dumps({"runs": results, "best": best}, indent=2) + "\n")
    np.savez_compressed(output / "splits_and_genes.npz", gene_indices=metadata["gene_indices"], **metadata["splits"])
    return {"runs": len(results), "best": best}


def run_retrain(config: dict, output: Path) -> dict:
    search_file = output / "search_results.json"
    if not search_file.exists(): raise FileNotFoundError("Run --phase search before retrain.")
    search = json.loads(search_file.read_text()); best = search["best"]
    paths = config["paths"]; cache, tokens = np.load(paths["embeddings"]), np.load(paths["tokens"])
    decoder_data = torch.load(paths["decoder"], map_location="cpu", weights_only=False)
    external = config["data"]["external_sources"]
    data_dir = Path(config["paths"].get("data_dir", "data"))
    source_paths = {name: data_dir / name for name in external} | {"E85.h5ad": data_dir / "E85.h5ad", "E95.h5ad": data_dir / "E95.h5ad"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    loss_cfg = selected["loss"]; loss_name = best["loss"]
    examples, holdout, _ = build_tasks(cache=cache, tokens=tokens, decoder_data=decoder_data, source_paths=source_paths,
        sequence=external, holdout_source=config["data"]["temporal_holdout"], official_sources=("E85.h5ad", "E95.h5ad"),
        official_holdout=0, ranking_genes=loss_cfg["ranking_genes"], loss_cells=config["training"]["loss_cells"],
        seed=config["seed"], device=device)
    holdout["normalized_target"] = normalized_target(holdout["target"], *holdout["normalization"], device)
    examples.append(holdout)
    # The real temporal validation in ``run_search`` selected this epoch budget.
    # Once all allowed data are included, retrain for that fixed budget without
    # pretending that the epoch index is a validation metric.
    epochs = max(1, int(selected["epoch"]) + 1); training = config["training"] | {"epochs": epochs, "patience": epochs + 1}
    validation: dict = {}
    final_dir = output / "final_retrain"
    result = train_run(examples=examples, validation=validation, architecture=best["architecture"], loss_config=loss_cfg,
        training=training, output=final_dir, seed=config["seed"], device=device, checkpoint_mode="last")
    manifest = {"checkpoint": str(result.checkpoint), "epochs": epochs, "uses_all_official_cells": True,
                "architecture": best["architecture"], "loss": loss_name,
                "epoch_selection": "best temporal-validation epoch from search",
                "checkpoint_policy": "fixed-budget final epoch; no fabricated validation score"}
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_generate(config: dict, output: Path) -> dict:
    manifest_path = output / "final_retrain" / "manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError("Run --phase retrain before generate.")
    manifest = json.loads(manifest_path.read_text()); paths = config["paths"]
    cache, tokens = np.load(paths["embeddings"]), np.load(paths["tokens"])
    external = config["data"]["external_sources"]
    external_states = [state_for_source(cache, tokens, name) for name in external]
    official = [state_for_source(cache, tokens, name) for name in ("E85.h5ad", "E95.h5ad")]
    center_mean, center_scale, dispersion_scale = scales(external_states + official)
    features, group_ids = token_features(official, [8.5, 9.5], 10.5, center_mean, center_scale, dispersion_scale)
    architecture = manifest["architecture"]; groups = len(official[-1]["proportion"]); latent_dim = official[-1]["latent_mean"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PopulationTransformer(features.shape[1], latent_dim, groups, architecture["model_dim"], architecture["heads"],
                                  architecture["layers"], architecture.get("dropout", .05)).to(device)
    checkpoint = torch.load(manifest["checkpoint"], map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"]); model.eval()
    with torch.no_grad(): raw_tokens = model(torch.tensor(features[None], device=device), torch.tensor(group_ids[None], device=device))
    proportion = raw_tokens["proportion"][0].cpu().numpy(); centers = raw_tokens["latent_mean"][0].cpu().numpy() * center_scale + center_mean
    dispersion = raw_tokens["latent_dispersion"][0].cpu().numpy() * dispersion_scale
    rng = np.random.default_rng(config["seed"]); n_cells = config["generation"]["n_cells"]
    counts = rng.multinomial(n_cells, proportion); generated_groups = np.repeat(np.arange(groups), counts); rng.shuffle(generated_groups)
    source_mask = cache["sources"] == "E95.h5ad"; source_latent = cache["latent"][source_mask]; source_groups = tokens["labels"][source_mask]
    selected, fallback = choose_residual_indices(generated_groups, source_groups, source_latent, tokens["centers"], config["seed"])
    latent_residual = source_latent[selected] - official[-1]["latent_mean"][source_groups[selected]]
    scale = dispersion[generated_groups] / np.maximum(official[-1]["latent_dispersion"][source_groups[selected]], 1e-4)
    generated_latent = (centers[generated_groups] + latent_residual * np.clip(scale, 0, 3)).astype(np.float32)
    decoder_data = torch.load(paths["decoder"], map_location="cpu", weights_only=False)
    decoder = NonnegativeLinearDecoder(decoder_data["input_dim"], decoder_data["output_dim"]); decoder.load_state_dict(decoder_data["state_dict"])
    raw_expression = decode(decoder, generated_latent, config["preparation"]["embedding_batch_size"], device)
    decoded_source = decode(decoder, source_latent[selected], config["preparation"]["embedding_batch_size"], device)
    data_dir = Path(paths.get("data_dir", "data"))
    source_expression, genes = read_rows(data_dir / "E95.h5ad", selected)
    corrected = np.maximum(raw_expression + source_expression - decoded_source, 0).astype(np.float32)
    nearest_indices = NearestNeighbors(n_neighbors=1).fit(source_latent).kneighbors(generated_latent, return_distance=False).ravel()
    nearest, nearest_genes = read_rows(data_dir / "E95.h5ad", nearest_indices)
    if not np.array_equal(genes, nearest_genes): raise ValueError("Official gene order mismatch.")
    generation_dir = output / "generation"; generation_dir.mkdir(parents=True, exist_ok=True); contracts = {}
    variants = {"decoder_raw": raw_expression, "expression_residual_e95": corrected, "nearest_e95": nearest}
    for name, matrix in variants.items():
        item = ad.AnnData(X=np.asarray(matrix, dtype=np.float32)); item.var_names = genes
        item.obs_names = [f"E020_{index:04d}" for index in range(n_cells)]
        item.uns.update(experiment="E020", seed=config["seed"], variant=name, sources=["D001", "D002", "D003-D011", "M001"],
                        target_stage="E10.5", target_data_used=False, celltype_required=False)
        contracts[name] = validate_task1_output(item, genes, min_cells=2500, max_cells=2500)
        item.write_h5ad(generation_dir / f"e10_5_{name}_2500.h5ad", compression="gzip")
    result = {"contracts": contracts, "fallback_cells": int(fallback), "primary": "expression_residual_e95"}
    (generation_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n"); return result


def run_prepare(config: dict) -> dict:
    """Regenerate portable decoder/embedding/token artifacts under outputs/."""
    paths = config["paths"]; data_dir = Path(paths.get("data_dir", "data")); cache_dir = Path(paths["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    decoder = Path(paths["decoder"])
    if not decoder.exists():
        command = [sys.executable, "-m", "src.scripts.run_scgpt_roundtrip", "--e85", str(data_dir / "E85.h5ad"),
            "--e95", str(data_dir / "E95.h5ad"), "--output-dir", str(cache_dir / "decoder_training"),
            "--cells-per-stage", str(config["preparation"]["decoder_cells_per_stage"]), "--input-genes",
            str(config["preparation"]["encoder_genes"]), "--epochs", str(config["preparation"]["decoder_epochs"]),
            "--covariance-weight", "0.1", "--covariance-genes", "128", "--torch-threads",
            str(config["training"]["threads"]), "--seed", str(config["seed"])]
        subprocess.run(command, check=True)
        trained = cache_dir / "decoder_training" / "decoder.pt"
        if not trained.exists(): raise FileNotFoundError(f"Decoder training did not create {trained}")
        decoder.write_bytes(trained.read_bytes())
    if not Path(paths["embeddings"]).exists() or not Path(paths["tokens"]).exists():
        inputs = [str(data_dir / name) for name in config["data"]["external_sources"]] + [str(data_dir / "E85.h5ad"), str(data_dir / "E95.h5ad")]
        command = [sys.executable, "-m", "src.scripts.run_m2_population_tokens", "--inputs", *inputs,
            "--output-dir", str(cache_dir), "--k", "50", "--seeds", str(config["seed"]), str(config["seed"] + 1),
            "--input-genes", str(config["preparation"]["encoder_genes"]), "--embedding-batch-size",
            str(config["preparation"]["embedding_batch_size"]), "--threads", str(config["training"]["threads"]),
            "--selected-genes-from", str(decoder)]
        subprocess.run(command, check=True)
    return {"decoder": str(decoder), "embeddings": paths["embeddings"], "tokens": paths["tokens"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--phase", choices=["all", "prepare", "search", "retrain", "generate"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config = load_config(args.config, args.smoke_test); validate_inputs(config, smoke=args.smoke_test)
    output = Path(config["paths"]["output_dir"])
    if args.smoke_test:
        print(json.dumps(run_smoke(config, output / "smoke"), indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if args.phase == "prepare":
        print(json.dumps(run_prepare(config), indent=2)); return
    if args.phase == "search":
        print(json.dumps(run_search(config, output, args.resume), indent=2)); return
    if args.phase == "retrain":
        print(json.dumps(run_retrain(config, output), indent=2)); return
    if args.phase == "generate":
        print(json.dumps(run_generate(config, output), indent=2)); return
    if args.phase == "all":
        run_prepare(config); run_search(config, output, args.resume); run_retrain(config, output)
        print(json.dumps(run_generate(config, output), indent=2)); return


if __name__ == "__main__":
    main()
