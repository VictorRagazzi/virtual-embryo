import numpy as np

from src.scripts.run_e016_expression_residual import choose_residual_indices


def test_residual_sampling_prefers_group_and_is_reproducible():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    source_groups = np.array([0, 0, 2])
    source_latent = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0]], dtype=np.float32)
    targets = np.array([0, 0, 1, 2])
    first, fallback = choose_residual_indices(targets, source_groups, source_latent, centers, 42)
    second, _ = choose_residual_indices(targets, source_groups, source_latent, centers, 42)
    assert np.array_equal(first, second)
    assert np.all(source_groups[first[:2]] == 0)
    assert source_groups[first[3]] == 2
    assert fallback == 1
