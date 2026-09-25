"""Training/search utilities for the artifact-backed E020 pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.approaches.population_transformer.objectives import (EarlyStopping, RFFMMD, de_ranking_loss,
    differentiable_population, selection_score)
from src.approaches.population_transformer.temporal import PopulationTransformer, copy_last, token_features
from src.scripts.run_e018_population_transformer import normalized_target, predict, scales


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def structural_loss(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
                    weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = target["proportion"] > 0
    pieces = {
        "proportion": (prediction["proportion"] - target["proportion"]).square().mean(),
        "center": (prediction["latent_mean"][:, mask] - target["latent_mean"][mask]).square().mean(),
        "dispersion": (prediction["latent_dispersion"][:, mask] - target["latent_dispersion"][mask]).square().mean(),
    }
    return sum(weights[name] * value for name, value in pieces.items()), pieces


def fixed_population_draw(proportion: np.ndarray, n_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(proportion, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("Population proportions must be finite, nonnegative and have positive mass.")
    probabilities /= probabilities.sum(dtype=np.float64)
    # NumPy checks sum(pvals[:-1]) before drawing. Put the largest mass last
    # and define it as the exact float64 remainder to avoid float32 roundoff.
    order = np.concatenate([np.flatnonzero(np.arange(len(probabilities)) != probabilities.argmax()),
                            np.asarray([probabilities.argmax()])])
    ordered = probabilities[order]
    ordered[-1] = max(0.0, 1.0 - ordered[:-1].sum(dtype=np.float64))
    ordered /= ordered.sum(dtype=np.float64)
    ordered[-1] = 1.0 - ordered[:-1].sum(dtype=np.float64)
    ordered_counts = rng.multinomial(n_cells, ordered)
    counts = np.empty_like(ordered_counts)
    counts[order] = ordered_counts
    groups = np.repeat(np.arange(len(proportion)), counts).astype(np.int64)
    return groups[rng.permutation(len(groups))]


def fixed_residual_draw(latent: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                        source_state: dict[str, np.ndarray], centers: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed); available = np.unique(labels); result = []; fallback = 0
    for group in groups:
        pool = np.flatnonzero(labels == group)
        if not len(pool):
            nearest = available[np.argmin(((centers[available] - centers[group]) ** 2).sum(1))]
            pool = np.flatnonzero(labels == nearest); fallback += 1
        index = int(rng.choice(pool))
        result.append(latent[index] - source_state["latent_mean"][labels[index]])
    return np.asarray(result, dtype=np.float32), fallback


def metric_errors(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], predicted_expression: np.ndarray,
                  target_expression: np.ndarray, reference_expression: np.ndarray, rff: RFFMMD,
                  temperature: float, top_k: int, block_size: int) -> dict[str, float]:
    observed = target["proportion"] > 0
    with torch.no_grad():
        pred, truth, ref = map(torch.as_tensor, (predicted_expression, target_expression, reference_expression))
        de = de_ranking_loss(pred, truth, ref, temperature=temperature, top_k=top_k, block_size=block_size)
        mmd = rff(pred, truth)
    center = np.square(prediction["latent_mean"][observed] - target["latent_mean"][observed]).mean()
    dispersion = np.square(prediction["latent_dispersion"][observed] - target["latent_dispersion"][observed]).mean()
    return {"de_ranking": float(de), "rff_mmd": float(mmd), "structure": float(center + dispersion),
            "proportion": float(np.square(prediction["proportion"] - target["proportion"]).mean())}


@dataclass
class TrainingResult:
    best_score: float
    best_epoch: int
    history: list[dict[str, float]]
    checkpoint: Path


def train_run(*, examples: list[dict], validation: dict, architecture: dict, loss_config: dict,
              training: dict, output: Path, seed: int, resume: Path | None = None,
              device: torch.device = torch.device("cpu"), checkpoint_mode: str = "validation") -> TrainingResult:
    """Train one Transformer and early-stop on relative gain against copy-last."""
    if checkpoint_mode not in {"validation", "last"}:
        raise ValueError("checkpoint_mode must be 'validation' or 'last'")
    seed_everything(seed, int(training.get("threads", 1))); output.mkdir(parents=True, exist_ok=True)
    feature_dim = examples[0]["features"].shape[1]; latent_dim = examples[0]["target"]["latent_mean"].shape[1]
    groups = len(examples[0]["target"]["proportion"])
    model = PopulationTransformer(feature_dim, latent_dim, groups, architecture["model_dim"], architecture["heads"],
                                  architecture["layers"], architecture.get("dropout", .05)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    amp_enabled = bool(training.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    stopper = EarlyStopping(int(training["patience"])); start = 0; history = []
    if resume and resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"]); stopper = EarlyStopping.from_state_dict(saved["early_stopping"])
        if "scaler" in saved: scaler.load_state_dict(saved["scaler"])
        start, history = int(saved["epoch"]) + 1, saved["history"]
    rff = RFFMMD(loss_config["ranking_genes"], loss_config["rff_features"], loss_config["rff_sigma"], seed).to(device)
    checkpoint = output / "checkpoint.pt"; best_epoch = start - 1
    for epoch in range(start, int(training["epochs"])):
        model.train(); totals = []
        for item in examples:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                raw = model(torch.as_tensor(item["features"][None], device=device), torch.as_tensor(item["groups"][None], device=device))
                target = {key: value.to(device) for key, value in item["normalized_target"].items()}
                loss, _ = structural_loss(raw, target, loss_config["structural_weights"])
                center_mean, center_scale, dispersion_scale = item.get("normalization", (0., 1., 1.))
                actual_centers = raw["latent_mean"][0] * center_scale + center_mean
                actual_dispersion = raw["latent_dispersion"][0] * dispersion_scale
                expression = differentiable_population(actual_centers, actual_dispersion,
                    item["draw_groups"].to(device), item["residuals"].to(device), item["decoder_weight"].to(device), item["decoder_bias"].to(device))
                if loss_config["ranking_weight"]:
                    loss = loss + loss_config["ranking_weight"] * de_ranking_loss(expression, item["target_expression"].to(device),
                        item["reference_expression"].to(device), temperature=loss_config["ranking_temperature"],
                        top_k=loss_config["top_k"], block_size=loss_config["ranking_block_size"])
                if loss_config["mmd_weight"]: loss = loss + loss_config["mmd_weight"] * rff(expression, item["target_expression"].to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); totals.append(float(loss.detach()))
        if checkpoint_mode == "validation":
            score = float(validation["score_fn"](model, rff))
            improved = stopper.update(score)
        else:
            # The epoch budget was selected on an independent temporal holdout.
            # Full-data retraining has no honest validation target, so save the
            # final epoch explicitly instead of fabricating a validation score.
            score = None
            improved = True
        row = {"epoch": epoch, "train_loss": float(np.mean(totals)), "validation_score": score}; history.append(row)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "early_stopping": stopper.state_dict(),
                 "epoch": epoch, "history": history, "architecture": architecture, "loss": loss_config, "seed": seed}
        torch.save(state, output / "last.pt")
        if improved: torch.save(state, checkpoint); best_epoch = epoch
        if checkpoint_mode == "validation" and stopper.stopped: break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    best_score = stopper.best if checkpoint_mode == "validation" else float("nan")
    return TrainingResult(best_score, best_epoch, history, checkpoint)
