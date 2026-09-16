"""Busca sequencial de hiperparâmetros para E8.5 → E9.5."""

import argparse
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import traceback

import torch

from .experiment import evaluate
from .temporal import train


# A ordem permite encerrar cedo com --max-experiments. Cada item é um treino.
SEARCH_CONFIGS = [
    {"learning_rate": 1e-4, "trainable_layers": 2, "expression_input": "none", "mmd_weight": 0.0},
    {"learning_rate": 3e-5, "trainable_layers": 2, "expression_input": "none", "mmd_weight": 0.0},
    {"learning_rate": 1e-4, "trainable_layers": 2, "expression_input": "none", "mmd_weight": 0.1},
    {"learning_rate": 1e-4, "trainable_layers": 2, "expression_input": "projected", "mmd_weight": 0.0},
    {"learning_rate": 1e-4, "trainable_layers": 1, "expression_input": "none", "mmd_weight": 0.0},
    {"learning_rate": 1e-4, "trainable_layers": 2, "expression_input": "projected", "mmd_weight": 0.1},
]

# O scorer trata DE como sinal principal de mudança. As distâncias desempatarão
# resultados de DE iguais na precisão publicada pelo Veckit.
CRITERION = "lexicográfico: maximizar de_score, maximizar de_direction, minimizar mmd_u, minimizar variogram"


def selection_key(metrics):
    values = []
    for name in ("de_score", "de_direction", "mmd_u", "variogram"):
        value = metrics.get(name)
        if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
            raise ValueError(f"Métrica {name} ausente ou não finita: {value!r}")
        values.append(float(value))
    return (values[0], values[1], -values[2], -values[3])


def save_results(path, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def run_search(args):
    if not 1 <= args.max_experiments <= len(SEARCH_CONFIGS):
        raise ValueError(f"max-experiments deve estar entre 1 e {len(SEARCH_CONFIGS)}.")
    result_dir = args.results_dir or args.output / "search" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Diretório de resultados já contém arquivos: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best.pt"
    if best_path.exists():
        shutil.copy2(best_path, result_dir / "previous_best.pt")

    records = []
    best_record = None
    best_key = None
    results_path = result_dir / "results.json"
    for number, config in enumerate(SEARCH_CONFIGS[:args.max_experiments], 1):
        iteration_dir = result_dir / f"iteration_{number:02d}"
        parameters = {"epochs": args.epochs, "batch_size": args.batch_size,
                      "max_cells": args.max_cells, "max_length": args.max_length,
                      "components": args.components, "validation_fraction": args.validation_fraction,
                      "seed": args.seed, "alpha": args.alpha, **config}
        record = {"iteration": number, "hyperparameters": parameters,
                  "criterion": CRITERION, "status": "running"}
        print(f"\n[{number}/{args.max_experiments}] Treino: {json.dumps(parameters)}", flush=True)
        try:
            training_args = argparse.Namespace(
                manifest=args.manifest, encoder=args.encoder, tokens=args.tokens,
                medians=args.medians, gene_map=args.gene_map,
                expression_scale=args.expression_scale, layer=args.layer,
                group_column=args.group_column, output=iteration_dir / "train",
                device=args.device, **{key: value for key, value in parameters.items() if key != "alpha"})
            train(training_args)
            checkpoint = training_args.output / "best.pt"
            result = evaluate(checkpoint, iteration_dir / "evaluation",
                              batch_size=args.batch_size, device=args.device, alpha=args.alpha)
            metrics = result["metrics"]
            key = selection_key(metrics)
            record.update({"status": "ok", "metrics": metrics, "meta": result.get("meta", {}),
                           "selection_key": key, "checkpoint": str(checkpoint)})
            record.update({name: metrics[name] for name in
                           ("de_score", "de_direction", "mmd_u", "variogram")})
            if best_key is None or key > best_key:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = best_path.with_name(f".best-{os.getpid()}.tmp")
                try:
                    shutil.copy2(checkpoint, temporary)
                    temporary.replace(best_path)
                finally:
                    temporary.unlink(missing_ok=True)
                best_key = key
                best_record = record
                record["new_best"] = True
                print(f"[{number}/{args.max_experiments}] Novo melhor modelo: {best_path}", flush=True)
            else:
                record["new_best"] = False
                print(f"[{number}/{args.max_experiments}] Concluído; melhor ainda é a iteração {best_record['iteration']}.", flush=True)
        except Exception as error:
            record.update({"status": "error", "error": str(error),
                           "traceback": traceback.format_exc()})
            print(f"[{number}/{args.max_experiments}] Falhou: {error}", flush=True)
        finally:
            records.append(record)
            save_results(results_path, records)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_record is None:
        raise RuntimeError(f"Nenhuma configuração foi avaliada com sucesso. Erros em {results_path}")
    print("\n=== Melhor configuração ===", flush=True)
    print(f"Iteração: {best_record['iteration']}", flush=True)
    print(f"Hiperparâmetros: {json.dumps(best_record['hyperparameters'])}", flush=True)
    for name in ("de_score", "de_direction", "mmd_u", "variogram"):
        print(f"{name}: {best_record[name]}", flush=True)
    print(f"Critério: {CRITERION}", flush=True)
    print(f"Modelo: {best_path}", flush=True)
    print(f"Resultados: {results_path}", flush=True)
    return best_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/T2/manifest.json"))
    parser.add_argument("--encoder", type=Path, default=Path("models/mouse-Geneformer"))
    parser.add_argument("--tokens", type=Path, default=Path("models/mouse-Geneformer/MLM-re_token_dictionary_v1.pkl"))
    parser.add_argument("--medians", type=Path, default=Path("models/mouse-Geneformer/mouse_gene_median_dictionary.pkl"))
    parser.add_argument("--gene-map", type=Path, default=Path("models/mouse-Geneformer/gene_map.csv"))
    parser.add_argument("--expression-scale", choices=["counts", "log1p"], default="log1p")
    parser.add_argument("--layer")
    parser.add_argument("--group-column")
    parser.add_argument("--output", type=Path, default=Path("models/T2_temporal"),
                        help="Diretório padrão do train, onde o melhor best.pt ficará disponível")
    parser.add_argument("--results-dir", type=Path, help="Novo diretório para os resultados desta busca")
    parser.add_argument("--max-experiments", type=int, default=len(SEARCH_CONFIGS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA não está disponível; execute no servidor com GPU.")
    run_search(args)


if __name__ == "__main__":
    main()
