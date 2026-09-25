"""E018: treina Transformer e MLP pequenos e avalia holdout temporal E8.5."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.approaches.population_transformer.temporal import (PopulationMLP, PopulationTransformer,
    state_for_source, token_features, token_metrics, validate_token_state)


EXTERNAL_SOURCES = ["E65_ex.h5ad", "E675_ex.h5ad", "E70_ex.h5ad", "E725_ex.h5ad", "E75_ex.h5ad",
                    "E775_ex.h5ad", "E80_ex.h5ad", "E825_ex.h5ad", "E85_ex.h5ad"]


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def scales(states: list[dict[str, np.ndarray]]) -> tuple[float, float, float]:
    centers = np.concatenate([s["latent_mean"][s["proportion"] > 0].ravel() for s in states])
    dispersions = np.concatenate([s["latent_dispersion"][s["proportion"] > 0].ravel() for s in states])
    return float(centers.mean()), float(centers.std() + 1e-6), float(dispersions.std() + dispersions.mean() + 1e-6)


def normalized_target(state: dict[str, np.ndarray], center_mean: float, center_scale: float,
                      dispersion_scale: float, device: torch.device) -> dict[str, torch.Tensor]:
    return {"proportion": torch.tensor(state["proportion"], device=device),
            "latent_mean": torch.tensor((state["latent_mean"] - center_mean) / center_scale, device=device),
            "latent_dispersion": torch.tensor(state["latent_dispersion"] / dispersion_scale, device=device)}


def loss_value(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    mask = target["proportion"] > 0
    prop = torch.mean((prediction["proportion"] - target["proportion"]) ** 2)
    center = torch.mean((prediction["latent_mean"][:, mask] - target["latent_mean"][mask]) ** 2)
    dispersion = torch.mean((prediction["latent_dispersion"][:, mask] - target["latent_dispersion"][mask]) ** 2)
    total = prop + center + dispersion
    return total, {"proportion": float(prop.detach()), "center": float(center.detach()), "dispersion": float(dispersion.detach())}


def predict(model: torch.nn.Module, features: np.ndarray, groups: np.ndarray, center_mean: float, center_scale: float,
            dispersion_scale: float, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(features[None], device=device), torch.tensor(groups[None], device=device))
    result = {"proportion": raw["proportion"][0].cpu().numpy().astype(np.float32),
              "latent_mean": (raw["latent_mean"][0].cpu().numpy() * center_scale + center_mean).astype(np.float32),
              "latent_dispersion": (raw["latent_dispersion"][0].cpu().numpy() * dispersion_scale).astype(np.float32)}
    validate_token_state(result)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--model-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed, args.threads); args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, tokens = np.load(args.cache), np.load(args.tokens)
    states = [state_for_source(cache, tokens, source) for source in EXTERNAL_SOURCES]
    times = [float(np.unique(cache["stages"][cache["sources"] == source]).item()[1:]) for source in EXTERNAL_SOURCES]
    train_states, holdout = states[:-1], states[-1]
    center_mean, center_scale, dispersion_scale = scales(train_states)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    examples = []
    for target_index in range(2, len(train_states)):
        features, groups = token_features(train_states[target_index-2:target_index], times[target_index-2:target_index],
                                          times[target_index], center_mean, center_scale, dispersion_scale)
        examples.append((features, groups, train_states[target_index]))
    holdout_features, holdout_groups = token_features(train_states[-2:], times[-3:-1], times[-1], center_mean, center_scale, dispersion_scale)
    feature_dim, latent_dim, n_groups = holdout_features.shape[1], holdout["latent_mean"].shape[1], len(holdout["proportion"])
    models = {"transformer": PopulationTransformer(feature_dim, latent_dim, n_groups, args.model_dim, args.heads, args.layers, args.dropout),
              "mlp": PopulationMLP(feature_dim, latent_dim, n_groups, args.model_dim)}
    all_metrics, histories, predictions = {}, {}, {}
    for model_index, (name, model) in enumerate(models.items()):
        set_seed(args.seed + model_index, args.threads); model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        history = []
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for features, groups, target_state in examples:
                optimizer.zero_grad()
                output = model(torch.tensor(features[None], device=device), torch.tensor(groups[None], device=device))
                target = normalized_target(target_state, center_mean, center_scale, dispersion_scale, device)
                loss, _ = loss_value(output, target); loss.backward(); optimizer.step(); epoch_loss += float(loss.detach())
            history.append(epoch_loss / len(examples))
        result = predict(model, holdout_features, holdout_groups, center_mean, center_scale, dispersion_scale, device)
        predictions[name] = result; all_metrics[name] = token_metrics(result, holdout) | {"final_train_loss": history[-1]}
        histories[name] = history
        torch.save({"state_dict": model.state_dict(), "model": name, "feature_dim": feature_dim, "latent_dim": latent_dim,
                    "groups": n_groups, "normalization": [center_mean, center_scale, dispersion_scale], "seed": args.seed,
                    "source_status": "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"}, args.output_dir / f"{name}.pt")
    np.savez_compressed(args.output_dir / "predictions.npz", **{
        f"{method}_{field}": value[field] for method, value in predictions.items() for field in value})
    config = vars(args) | {"device_used": str(device), "training_sources": EXTERNAL_SOURCES[:-1],
                           "holdout_source": EXTERNAL_SOURCES[-1], "training_examples": len(examples),
                           "normalization_training_only": [center_mean, center_scale, dispersion_scale],
                           "vocabulary_leakage": "K=50 fit jointly including holdout and official E8.5/E9.5",
                           "source_status": "USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    (args.output_dir / "training_history.json").write_text(json.dumps(histories, indent=2) + "\n")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__": main()
