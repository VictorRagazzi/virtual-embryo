"""Treino multitemporal e previsão em h5ad."""

import argparse
import itertools
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (load_dictionary, read_gene_map, read_stage,
                   pair_cells, split_cells)
from .model import TemporalGeneformer, load_pretrained_encoder


class TemporalPairs(Dataset):
    def __init__(self, stages, pairs):
        self.stages = stages
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        start, end, source, target = self.pairs[index]
        stage = self.stages[start]
        return (torch.from_numpy(stage["input_ids"][source]),
                torch.from_numpy(stage["mask"][source]),
                torch.from_numpy(stage["expression"][source].toarray().ravel()),
                torch.tensor([start, end - start], dtype=torch.float32),
                torch.from_numpy(self.stages[end]["expression"][target].toarray().ravel()))


def build_pairs(stages, splits, components, seed):
    """Ajusta coordenadas só no treino; busca destinos dentro do mesmo split."""
    training = sparse.vstack([stage["expression"][splits[day][0]] for day, stage in stages.items()])
    if components < 1:
        raise ValueError("components deve ser positivo.")
    if training.shape[1] < 2:
        raise ValueError("São necessários pelo menos dois genes comuns.")
    projection = TruncatedSVD(n_components=min(components, training.shape[0] - 1, training.shape[1] - 1), random_state=seed)
    projection.fit(training)
    coordinates = {day: projection.transform(stage["expression"]) for day, stage in stages.items()}
    pairs = [[], []]
    records = []
    for start, end in itertools.combinations(sorted(stages), 2):
        for split, label in enumerate(["train", "validation"]):
            source_indices, target_indices = splits[start][split], splits[end][split]
            neighbors, distances = pair_cells(coordinates[start][source_indices], coordinates[end][target_indices])
            for source, target, distance in zip(source_indices, target_indices[neighbors], distances):
                pairs[split].append((start, end, int(source), int(target)))
                records.append({"split": label, "time": start, "future_time": end,
                                "source_cell": str(stages[start]["obs"].index[source]),
                                "target_cell": str(stages[end]["obs"].index[target]),
                                "distance": float(distance)})
    return pairs, pd.DataFrame(records)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch_size e learning_rate devem ser positivos.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Diretório de saída deve estar vazio: {args.output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokens = {str(gene): int(value) for gene, value in load_dictionary(args.tokens).items()}
    medians = {str(gene): float(value) for gene, value in load_dictionary(args.medians).items()}
    gene_map = read_gene_map(args.gene_map)
    encoder = load_pretrained_encoder(args.encoder)
    if args.max_length > encoder.config.max_position_embeddings:
        raise ValueError("max_length excede as posições do encoder.")
    if max(tokens.values()) >= encoder.config.vocab_size or min(tokens.values()) < 0:
        raise ValueError("Vocabulário incompatível com o encoder.")
    if tokens["<pad>"] != encoder.config.pad_token_id:
        raise ValueError("Token de padding difere do checkpoint.")
    manifest = json.loads(args.manifest.read_text())
    excluded = {cell for sample in manifest["evaluation"].values() for cell in sample["cells"]}
    stages = {}
    for specification in manifest["stages"]:
        day_text, path = specification.split("=", 1)
        day = float(day_text)
        if not np.isfinite(day) or day in stages:
            raise ValueError("Estágios devem ser finitos e distintos.")
        stages[day] = read_stage(path, args.expression_scale, tokens, medians,
                                 args.max_length, gene_map, args.layer, args.max_cells, args.seed, excluded)
    if len(stages) < 2:
        raise ValueError("Informe pelo menos dois estágios.")
    first = stages[min(stages)]
    common = set.intersection(*(set(stage["genes"]) for stage in stages.values()))
    genes = [gene for gene in first["genes"] if gene in common]
    if len(genes) < 2:
        raise ValueError("Menos de dois genes comuns entre estágios.")
    splits = {}
    validation_groups = None
    if args.group_column:
        # Um mesmo embrião/réplica permanece no mesmo split em todos os tempos.
        observations = pd.concat([stage["obs"] for stage in stages.values()], ignore_index=True)
        _, validation_indices = split_cells(observations, args.validation_fraction, args.seed, args.group_column)
        validation_groups = set(observations.iloc[validation_indices][args.group_column])
    for day, stage in stages.items():
        positions = pd.Index(stage["genes"]).get_indexer(genes)
        stage["expression"] = stage["expression"][:, positions].tocsr()
        if validation_groups is None:
            splits[day] = split_cells(stage["obs"], args.validation_fraction, args.seed)
        else:
            validation = stage["obs"][args.group_column].isin(validation_groups).to_numpy()
            splits[day] = np.flatnonzero(~validation), np.flatnonzero(validation)
            if any(len(indices) == 0 for indices in splits[day]):
                raise ValueError(f"Estágio {day} sem células em um split; revise grupos/fração/seed.")
    pairs, records = build_pairs(stages, splits, args.components, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output / "pairs.csv", index=False)
    model = TemporalGeneformer(encoder, len(genes), args.mode, args.trainable_layers).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    loaders = [DataLoader(TemporalPairs(stages, subset), batch_size=args.batch_size, shuffle=(i == 0)) for i, subset in enumerate(pairs)]
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        metrics = {"epoch": epoch + 1}
        for split, loader in enumerate(loaders):
            model.train(split == 0)
            squared_error = baseline_error = count = 0
            for batch in loader:
                ids, mask, expression, times, target = [value.to(args.device) for value in batch]
                with torch.set_grad_enabled(split == 0):
                    prediction = model(ids, mask, expression, times)
                    # Mesmo clamp usado na exportação, apenas durante avaliação.
                    evaluated = prediction if split == 0 else prediction.clamp_min(0)
                    loss = torch.nn.functional.mse_loss(evaluated, target)
                    if split == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                squared_error += loss.item() * target.numel()
                baseline_error += (expression - target).square().sum().item()
                count += target.numel()
            label = "train" if split == 0 else "validation"
            metrics[f"{label}_mse"] = squared_error / count
            metrics[f"{label}_persistence_mse"] = baseline_error / count
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["validation_mse"] < best:
            best = metrics["validation_mse"]
            torch.save({"state_dict": model.state_dict(), "encoder_config": encoder.config.to_dict(),
                        "genes": genes, "tokens": tokens, "medians": medians, "gene_map": gene_map,
                        "expression_scale": args.expression_scale, "layer": args.layer,
                        "max_length": args.max_length, "mode": args.mode,
                        "trainable_layers": args.trainable_layers, "stages": sorted(stages),
                        "manifest": manifest}, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "function"}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def predict(args):
    from transformers import BertConfig, BertModel

    if not np.isfinite([args.time, args.delta_time]).all() or args.delta_time <= 0:
        raise ValueError("Tempo deve ser finito e delta_time deve ser positivo.")
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    stage = read_stage(args.input, checkpoint["expression_scale"], checkpoint["tokens"],
                       checkpoint["medians"], checkpoint["max_length"], checkpoint["gene_map"], checkpoint["layer"])
    positions = pd.Index(stage["genes"]).get_indexer(checkpoint["genes"])
    if (positions < 0).any():
        raise ValueError("Entrada não contém todos os genes de saída do treino.")
    expression = stage["expression"][:, positions].tocsr()
    encoder = BertModel(BertConfig.from_dict(checkpoint["encoder_config"]), add_pooling_layer=False)
    model = TemporalGeneformer(encoder, len(positions), checkpoint["mode"], checkpoint["trainable_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, expression.shape[0], args.batch_size):
            selection = slice(start, start + args.batch_size)
            current = torch.from_numpy(expression[selection].toarray()).to(args.device)
            ids = torch.from_numpy(stage["input_ids"][selection]).to(args.device)
            mask = torch.from_numpy(stage["mask"][selection]).to(args.device)
            times = torch.tensor([args.time, args.delta_time], device=args.device).float().expand(len(current), -1)
            predictions.append(model(ids, mask, current, times).clamp_min(0).cpu().numpy())
    observations = stage["obs"].copy()
    observations["source_time"] = args.time
    observations["timepoint"] = args.time + args.delta_time
    # Rótulos observados descrevem a origem, não um tipo futuro previsto.
    observations = observations.rename(columns={name: f"source_{name}" for name in ["celltype", "cell_type"] if name in observations})
    output = ad.AnnData(np.concatenate(predictions), obs=observations,
                       var=pd.DataFrame(index=checkpoint["genes"]))
    output.uns["expression_scale"] = "log1p_normalized_10000_before_gene_selection"
    output.uns["training_stages"] = checkpoint["stages"]
    output.uns["prediction_mode"] = checkpoint["mode"]
    output.uns["checkpoint"] = str(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output, compression="gzip")
    print(f"Previsão salva: {args.output} ({output.n_obs} células, {output.n_vars} genes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.set_defaults(function=train)
    training.add_argument("--manifest", type=Path, required=True, help="Manifesto criado por experiment prepare")
    training.add_argument("--encoder", type=Path, required=True)
    training.add_argument("--tokens", type=Path, required=True)
    training.add_argument("--medians", type=Path, required=True)
    training.add_argument("--gene-map", type=Path, help="CSV: gene_symbol,ensembl_id")
    training.add_argument("--expression-scale", choices=["counts", "log1p"], required=True)
    training.add_argument("--layer")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--mode", choices=["delta", "direct"], default="delta")
    training.add_argument("--trainable-layers", type=int, default=2)
    training.add_argument("--max-length", type=int, default=2048)
    training.add_argument("--max-cells", type=int, default=2000)
    training.add_argument("--components", type=int, default=50)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--group-column", help="Embrião/réplica, quando disponível em todos os estágios")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict")
    prediction.set_defaults(function=predict)
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--time", type=float, required=True)
    prediction.add_argument("--delta-time", type=float, required=True)
    for command in [training, prediction]:
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size deve ser positivo")
    if getattr(args, "max_cells", None) is not None and args.max_cells < 2:
        parser.error("max-cells deve ser pelo menos 2")
    args.function(args)


if __name__ == "__main__":
    main()
