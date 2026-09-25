import numpy as np

from src.scripts.run_m3_known_population import rff_mmd, sample_population


def test_residual_sampling_preserves_diversity_and_centroid_collapses():
    latent = np.array([[0., 0.], [1., 0.], [10., 0.], [12., 0.]], dtype=np.float32)
    stages = np.array(["E9.5"] * 4); labels = np.array([0, 0, 1, 1])
    residual, groups, fallback = sample_population(latent, stages, labels, "E9.5", 100, 42, True)
    centroid, _, _ = sample_population(latent, stages, labels, "E9.5", 100, 42, False)
    assert fallback == 0 and len(groups) == 100
    assert residual.var(axis=0).sum() > centroid.var(axis=0).sum()
    assert len(np.unique(centroid, axis=0)) == 2


def test_rff_mmd_is_zero_for_identical_arrays():
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    assert rff_mmd(x, x, 64, 42) < 1e-12
