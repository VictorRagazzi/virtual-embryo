import numpy as np

from src.approaches.population_transformer.anchored import fit_gene_programs, hybrid_gene_programs


def test_hybrid_preserves_general_basis_and_adds_orthogonal_capacity():
    rng = np.random.default_rng(42)
    changes = rng.normal(size=(40, 80)).astype(np.float32)
    pb = rng.normal(size=(6, 80)).astype(np.float32)
    _, general = fit_gene_programs(changes, 8, 42)
    hybrid = hybrid_gene_programs(changes, pb, 8, 8, 42)
    np.testing.assert_array_equal(hybrid[:8], general)
    np.testing.assert_allclose(hybrid @ hybrid.T, np.eye(16), atol=2e-6)
    np.testing.assert_array_equal(hybrid, hybrid_gene_programs(changes, pb, 8, 8, 42))
    old_error = np.linalg.norm(changes - changes @ general.T @ general)
    new_error = np.linalg.norm(changes - changes @ hybrid.T @ hybrid)
    assert new_error < old_error
