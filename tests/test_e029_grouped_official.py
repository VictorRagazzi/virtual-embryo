import torch

from src.approaches.population_transformer.nonlinear_pca import GroupTransitionTransformer


def test_group_transition_contract():
    torch.manual_seed(42)
    model = GroupTransitionTransformer(4, 3, 16, 4, 2, 0)
    state = torch.zeros(1, 3, 10)
    output = model(state, torch.arange(3)[None])
    assert output["latent_mean"].shape == (1, 3, 4)
    assert output["latent_dispersion"].shape == (1, 3, 4)
    assert torch.all(output["latent_dispersion"] > 0)
    assert torch.allclose(output["proportion"].sum(1), torch.ones(1))
