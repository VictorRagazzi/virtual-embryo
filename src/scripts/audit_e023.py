"""Audit E023 populations against the frozen E021 evaluation, without hidden data."""
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from src.m0_contract import validate_task1_output


def main():
    root = Path('outputs/population_transformer/E023')
    baseline = Path('outputs/population_transformer/E021/fixed_eval_seed42')
    truth = ad.read_h5ad(baseline / 'target.h5ad')
    reference = ad.read_h5ad(baseline / 'reference.h5ad')
    rows = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / 'metrics.json').exists():
            continue
        for name, expected in [('reference', reference), ('target', truth)]:
            value = ad.read_h5ad(directory / f'{name}.h5ad')
            assert np.array_equal(value.X, expected.X), name
            assert np.array_equal(value.var_names, expected.var_names), name
        prediction = ad.read_h5ad(directory / 'anchored_0.25.h5ad')
        validate_task1_output(prediction, truth.var_names.to_numpy(), min_cells=512, max_cells=512)
        programs = np.load(directory / 'programs.npz')['programs']
        config = json.loads((directory / 'config.json').read_text())
        if config['n_programs'] == 32 and config['de_programs'] == 0:
            original = ad.read_h5ad(baseline / 'anchored_0.25.h5ad')
            np.testing.assert_array_equal(prediction.X, original.X)
        entry = {
            'reference_target_exactly_equal_e021': True,
            'unique_fraction': len(np.unique(prediction.X, axis=0)) / prediction.n_obs,
            'zero_fraction': float((prediction.X == 0).mean()),
            'target_zero_fraction': float((truth.X == 0).mean()),
            'orthogonality_max_error': float(np.abs(programs @ programs.T - np.eye(len(programs))).max()),
            'sha256': {},
            'config': config,
        }
        for filename in ['anchored_0.25.h5ad', 'checkpoint.pt', 'programs.npz', 'config.json', 'history.json']:
            path = directory / filename
            entry['sha256'][filename] = {'hash': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        score_path = directory / 'veckit.json'
        if score_path.exists():
            entry['veckit'] = json.loads(score_path.read_text())['metrics']
        rows[directory.name] = entry
    (root / 'audit.json').write_text(json.dumps(rows, indent=2) + '\n')
    sources = [Path('src/approaches/population_transformer/anchored.py'),
               Path('src/scripts/run_e021_anchored.py'), Path('src/scripts/audit_e023.py'),
               Path('tests/test_e023_programs.py')]
    (root / 'code_hashes.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in sources}, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
