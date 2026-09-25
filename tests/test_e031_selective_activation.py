import numpy as np

from src.scripts.run_e031_selective_activation import prevalence_policy, selective_group_expression


def test_prevalence_policy_requires_target_support_gain_and_positive_mean():
    source = np.array([[0, 0, 2], [0, 1, 2]], dtype=np.float32)
    target = np.array([[1, 0, 1], [2, 1, 1]], dtype=np.float32)
    mask, confidence = prevalence_policy(source, np.zeros(2, int), target, np.zeros(2, int), 1, 0.5, 0.25)
    np.testing.assert_array_equal(mask, [[True, False, False]])
    assert confidence[0, 0] == 1


def test_selective_activation_preserves_ineligible_zeros_and_caps_new_genes():
    base = np.array([[0, 0, 1, 0]], dtype=np.float32)
    delta = np.array([[1, 1, 0.5, 1]], dtype=np.float32)
    mask = np.array([[True, True, False, True]])
    confidence = np.array([[0.2, 0.8, 0, 0.5]], dtype=np.float32)
    donors = np.array([[2, 3, 4, 5]], dtype=np.float32)
    result, stats = selective_group_expression(
        base, np.array([0]), delta, mask, confidence, donors, np.array([0]),
        expressed_scale=0.25, activation_scale=0.125, max_activations_per_cell=2, seed=42,
    )
    assert result[0, 0] == 0
    assert result[0, 1] > 0 and result[0, 3] > 0 and result[0, 2] > 0
    assert stats["total_activations"] == 2
    np.testing.assert_allclose(np.expm1(result).sum(1), 10_000, rtol=1e-6)
    assert result.dtype == np.float32 and np.isfinite(result).all()
