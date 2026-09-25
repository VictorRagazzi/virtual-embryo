"""Summarize fixed-evaluation stability without inspecting hidden targets."""
import json
from pathlib import Path

import anndata as ad
import numpy as np


def main():
    root = Path('outputs/population_transformer/E025')
    baseline = Path('outputs/population_transformer/E021/confirm_seed42_n1024')
    reference, target = [ad.read_h5ad(baseline / f'{name}.h5ad') for name in ['reference', 'target']]
    scores = {}
    for seed in [42, 43, 44]:
        directory = root / f'top250_seed{seed}_n1024'
        scores[str(seed)] = json.loads((directory / 'veckit_0.125.json').read_text())['metrics']
        if seed != 42:
            for name, expected in [('reference', reference), ('target', target)]:
                actual = ad.read_h5ad(root / f'base_seed{seed}_n1024' / f'{name}.h5ad')
                np.testing.assert_array_equal(actual.X, expected.X)
                np.testing.assert_array_equal(actual.var_names, expected.var_names)
    summary = {}
    for metric in ['de_score', 'de_direction', 'mmd_u', 'variogram', 'pb_rel_err', 'composition_JSD']:
        values = np.array([scores[str(seed)][metric] for seed in [42, 43, 44]])
        worst = np.argmin(values) if metric in ['de_score', 'de_direction'] else np.argmax(values)
        summary[metric] = dict(mean=float(values.mean()), std_population=float(values.std()),
                               worst=float(values[worst]), worst_seed=[42, 43, 44][worst])
    result = dict(seeds=scores, summary=summary, program_seed=42, evaluation_seed=42,
                  identical_reference_and_target=True)
    (root / 'stability.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
