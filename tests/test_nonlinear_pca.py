import numpy as np
import torch

from src.approaches.population_transformer.nonlinear_pca import (
    ApproximateKernelPCA, CellSetTransformer, decode_variants, decode_zero_weighted_change, distribution_loss)


def test_nonlinear_pca_decode_variants_are_valid_and_residual_recovers_source():
    rng = np.random.default_rng(42)
    x = np.maximum(rng.normal(size=(24, 12)), 0).astype(np.float32)
    model = ApproximateKernelPCA(12, 4, .2, 1e-3, 42).fit_embedding(x[:, :6])
    z = model.transform(x[:, :6]); model.fit_decoder([z], [x])
    variants = decode_variants(model, z[:5], z, x)
    assert set(variants) == {"decoder_raw", "decoder_residual", "nearest_neighbor"}
    for value in variants.values():
        assert value.shape == (5, 12) and value.dtype == np.float32
        assert np.isfinite(value).all() and (value >= 0).all()
    np.testing.assert_allclose(variants["decoder_residual"], x[:5], atol=2e-5)


def test_zero_weighted_change_shrinks_only_supported_anchor_zeros():
    class Decoder:
        def inverse_transform(self, latent):
            return np.asarray(latent, dtype=np.float32)

    result = decode_zero_weighted_change(
        Decoder(), np.array([[3, 3, 3]], np.float32), np.array([[1, 1, 1]], np.float32),
        np.array([[4, 0, 0]], np.float32), np.array([0]), np.array([[True, True, False]]), .2,
    )
    np.testing.assert_allclose(result, [[6, .4, 0]])


def test_cell_transformer_and_distribution_loss_do_not_require_pairs():
    torch.manual_seed(42)
    model = CellSetTransformer(4, 8, 2, 1, 0)
    first, second, target = (torch.randn(1, 6, 4) for _ in range(3))
    prediction = model(first, second)
    directions = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    loss = distribution_loss(prediction, target, directions)
    shuffled = distribution_loss(prediction, target[:, torch.randperm(6)], directions)
    assert prediction.shape == second.shape and torch.isfinite(loss)
    assert torch.allclose(loss, shuffled, atol=1e-6)
