"""Small gene-wise temporal correction to an existing anchored population."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from src.m0_contract import validate_task1_output


def sparse_shift(matrix, delta):
    """Return a nonnegative float32 copy shifted only at positive entries."""
    return np.maximum(matrix + np.where(matrix > 0, delta, 0), 0).astype(np.float32)


def strongest_changes(delta, k):
    """Keep the k largest absolute changes, resolving ties by gene order."""
    if not 0 <= k <= len(delta):
        raise ValueError('top_genes must be between zero and the gene count')
    if k == 0:
        return delta.copy()
    selected = np.argsort(-np.abs(delta), kind='stable')[:k]
    result = np.zeros_like(delta)
    result[selected] = delta[selected]
    return result


def pseudobulk(path):
    data = ad.read_h5ad(path)
    return np.asarray(data.X.mean(axis=0)).ravel().astype(np.float32), data.var_names.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weights', type=float, nargs='+', default=[.0625, .125, .25])
    parser.add_argument('--first', type=Path, default=Path('data/E80_ex.h5ad'))
    parser.add_argument('--previous', type=Path, default=Path('data/E825_ex.h5ad'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-genes', type=int, default=0)
    parser.add_argument('--experiment', default='E024')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    first, genes = pseudobulk(args.first)
    previous, second_genes = pseudobulk(args.previous)
    baseline = ad.read_h5ad(args.baseline / 'anchored_0.25.h5ad')
    baseline_config = json.loads((args.baseline / 'config.json').read_text())
    assert np.array_equal(genes, second_genes) and np.array_equal(genes, baseline.var_names)
    delta = previous - first
    delta = strongest_changes(delta, args.top_genes)
    np.save(args.output_dir / 'trend.npy', delta)
    results = {}
    for weight in args.weights:
        output = baseline.copy()
        output.X = sparse_shift(baseline.X, weight * delta)
        output.uns.update(experiment=args.experiment, trend_weight=weight, seed=args.seed)
        contract = validate_task1_output(output, genes, min_cells=baseline.n_obs, max_cells=baseline.n_obs)
        path = args.output_dir / f'trend_{weight:g}.h5ad'
        output.write_h5ad(path, compression='gzip')
        results[str(weight)] = dict(contract=contract, unique_cells=len(np.unique(output.X, axis=0)),
                                   zero_fraction=float((output.X == 0).mean()),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    config = vars(args) | {'transformer_seed': baseline_config['seed'],
                          'program_seed': baseline_config.get('program_seed', 42),
                          'evaluation_seed': args.seed,
                          'baseline_config': baseline_config}
    (args.output_dir / 'config.json').write_text(json.dumps(config, default=str, indent=2))
    (args.output_dir / 'contracts.json').write_text(json.dumps(results, default=str, indent=2))
    print(json.dumps(results, default=str, indent=2))


if __name__ == '__main__':
    main()
