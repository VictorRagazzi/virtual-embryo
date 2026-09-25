import numpy as np
from sklearn.cluster import KMeans

from src.scripts.run_m2_population_tokens import matched_proportion_l1, stage_from_path, summarize


def test_stage_from_path_supports_official_and_extra_names():
    from pathlib import Path
    assert stage_from_path(Path("E85.h5ad")) == "E8.5"
    assert stage_from_path(Path("E725_ex.h5ad")) == "E7.25"


def test_population_summary_preserves_empty_groups_and_contract():
    latent = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    programs = np.array([[1, 0], [2, 1], [4, 3]], dtype=np.float32)
    result = summarize(latent, programs, np.array(["E8.5", "E8.5", "E9.5"]), np.array([0, 0, 1]), 3, ["a", "b"])
    assert result["cell_count"].shape == (2, 3)
    assert np.allclose(result["proportion"].sum(axis=1), 1)
    assert result["empty_mask"].sum() == 4
    assert result["latent_mean"].shape == (2, 3, 2)
    assert result["gene_program_scores"].shape == (2, 3, 2)
    assert result["latent_residual"].shape == latent.shape
    assert np.allclose(result["latent_residual"].mean(axis=0), 0)


def test_matched_proportions_ignore_label_permutation():
    x = np.array([[0.0], [0.1], [10.0], [10.1]])
    a = KMeans(2, random_state=1, n_init=1).fit(x)
    b = KMeans(2, random_state=2, n_init=1).fit(x)
    assert matched_proportion_l1(a, b, a.labels_, b.labels_) == 0
