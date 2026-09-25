import numpy as np

from src.scripts.run_e029_generate import anchored_group_expression, normalize_log1p_cp10k
from src.scripts.run_e029_grouped_official import generate


def test_generation_requires_supported_groups_and_requested_count():
    class Decoder:
        def inverse_transform(self, latent):
            return np.maximum(np.pad(latent, ((0, 0), (0, 1))), 0).astype(np.float32)
    source_latent = np.array([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=np.float32)
    source_expression = np.pad(source_latent, ((0, 0), (0, 1))).astype(np.float32)
    state = {"proportion": np.array([.5, .5], np.float32), "latent_mean": np.array([[1.5, 0], [0, 1.5]], np.float32),
             "latent_dispersion": np.array([[.5, 0], [0, .5]], np.float32)}
    prediction = {key: value.copy() for key, value in state.items()}
    matrix, groups = generate(prediction, state, source_latent, source_expression, np.array([0, 0, 1, 1]), Decoder(), 3, 42)
    assert matrix.shape == (3, 3) and len(groups) == 3
    assert np.isfinite(matrix).all() and (matrix >= 0).all()


def test_cp10k_normalization_preserves_zeros_and_library_size():
    matrix = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.float32)
    result = normalize_log1p_cp10k(matrix)
    np.testing.assert_array_equal(result == 0, matrix == 0)
    np.testing.assert_allclose(np.expm1(result).sum(1), 10_000, rtol=1e-6)
    assert result.dtype == np.float32 and np.isfinite(result).all()


def test_anchored_group_expression_preserves_unsupported_zeros():
    base = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.float32)
    delta = np.array([[5, -.2, .4], [-1, 8, .2]], dtype=np.float32)
    result = anchored_group_expression(base, np.array([0, 1]), delta, .25)
    np.testing.assert_array_equal(result == 0, base == 0)
    np.testing.assert_allclose(np.expm1(result).sum(1), 10_000, rtol=1e-6)
