import numpy as np
import torch

from src.approaches.population_transformer.temporal import (PopulationTransformer, copy_last, linear_extrapolation,
                                                            state_for_source, validate_token_state)


def _state(offset=0.0):
    return {"proportion": np.array([0.25, 0.75], dtype=np.float32),
            "latent_mean": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32) + offset,
            "latent_dispersion": np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32) + offset}


def test_baselines_keep_shapes_mass_and_nonnegative_dispersion():
    first, second = _state(), _state(0.1)
    for result in (copy_last(second), linear_extrapolation(first, second, 1.0)):
        validate_token_state(result)
        assert result["latent_mean"].shape == (2, 2)
        assert np.isclose(result["proportion"].sum(), 1.0)
        assert np.all(result["latent_dispersion"] >= 0)


def test_linear_extrapolation_is_reproducible():
    first, second = _state(), _state(0.1)
    left = linear_extrapolation(first, second, 1.0)
    right = linear_extrapolation(first, second, 1.0)
    assert all(np.array_equal(left[key], right[key]) for key in left)


def test_source_state_keeps_masks_and_mass_separate():
    cache = {"sources": np.array(["a", "a", "b"]),
             "latent": np.array([[0., 1.], [2., 3.], [9., 9.]], dtype=np.float32)}
    tokens = {"labels": np.array([0, 0, 1]), "centers": np.zeros((2, 2), dtype=np.float32)}
    state = state_for_source(cache, tokens, "a")
    assert np.array_equal(state["proportion"], [1., 0.])
    assert np.array_equal(state["latent_mean"][1], [0., 0.])


def test_transformer_output_contract_and_reproducibility():
    torch.manual_seed(42)
    model = PopulationTransformer(feature_dim=7, latent_dim=2, groups=3, model_dim=8, heads=2, layers=1, dropout=0)
    features = torch.zeros(1, 6, 7)
    groups = torch.tensor([[0, 1, 2, 0, 1, 2]])
    first, second = model(features, groups), model(features, groups)
    assert first["latent_mean"].shape == (1, 3, 2)
    assert torch.allclose(first["proportion"].sum(1), torch.ones(1))
    assert torch.all(first["latent_dispersion"] > 0)
    assert all(torch.equal(first[key], second[key]) for key in first)
