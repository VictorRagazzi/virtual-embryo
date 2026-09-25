"""E017: avalia copy-last e extrapolação linear no holdout temporal E8.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.approaches.population_transformer.temporal import copy_last, linear_extrapolation, state_for_source, token_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-source", default="E80_ex.h5ad")
    parser.add_argument("--last-source", default="E825_ex.h5ad")
    parser.add_argument("--target-source", default="E85_ex.h5ad")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens, cache = np.load(args.tokens), np.load(args.cache)
    source_names = (args.first_source, args.last_source, args.target_source)
    first, last, target = (state_for_source(cache, tokens, source) for source in source_names)
    source_to_time = {source: float(np.unique(cache["stages"][cache["sources"] == source]).item()[1:]) for source in source_names}
    times = [source_to_time[source] for source in source_names]
    factor = (times[2] - times[1]) / (times[1] - times[0])
    predictions = {"copy_last": copy_last(last), "linear_extrapolation": linear_extrapolation(first, last, factor)}
    metrics = {name: token_metrics(value, target) for name, value in predictions.items()}
    np.savez_compressed(args.output_dir / "predictions.npz", **{
        f"{method}_{field}": value[field] for method, value in predictions.items() for field in value})
    config = vars(args) | {"times": times, "linear_factor": factor,
                           "vocabulary_leakage": "K=50 fit jointly including E85_ex holdout and official stages"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
