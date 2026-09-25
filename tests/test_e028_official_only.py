import torch

from src.approaches.population_transformer.nonlinear_pca import SingleStageCellTransformer, distribution_loss


def test_single_stage_transformer_has_population_output_and_finite_gradient():
    torch.manual_seed(42)
    model = SingleStageCellTransformer(4, 16, 4, 2, 0)
    source, target = torch.randn(1, 8, 4), torch.randn(1, 8, 4)
    directions = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
    prediction = model(source); loss = distribution_loss(prediction, target, directions)
    loss.backward()
    assert prediction.shape == source.shape
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
