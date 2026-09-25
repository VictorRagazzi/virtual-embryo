import numpy as np
import torch

from src.approaches.population_transformer.data_v2 import official_split, residual_indices
from src.approaches.population_transformer.objectives import (EarlyStopping, RFFMMD, de_ranking_loss,
    differentiable_population, hybrid_gene_selection, multicut_signed_de_loss, selection_score, soft_rank)


def test_official_split_reproducible_disjoint_and_stratified():
    labels = np.array(["a"] * 30 + ["b"] * 20)
    train, holdout = official_split(labels, 10, 42)
    train2, holdout2 = official_split(labels, 10, 42)
    assert np.array_equal(train, train2) and np.array_equal(holdout, holdout2)
    assert not np.intersect1d(train, holdout).size
    assert len(train) + len(holdout) == len(labels)
    assert dict(zip(*np.unique(labels[holdout], return_counts=True))) == {"a": 6, "b": 4}


def test_hybrid_selection_unique_and_train_statistics_only():
    selected = hybrid_gene_selection(np.array([9, 8, 1, 0]), np.array([0, 1, 8, 9]), 4)
    assert len(selected) == len(np.unique(selected)) == 4
    assert set(selected[:2]) == {0, 1}


def test_soft_rank_order_and_de_loss_finite_gradients():
    assert torch.allclose(soft_rank(torch.tensor([3., 2., 1.]), .05, 2), torch.tensor([1.5, 2.5, 3.5]), atol=.01)
    reference = torch.rand(8, 12)
    target = reference + torch.linspace(-.2, .2, 12)
    predicted = torch.nn.Parameter(reference.clone())
    loss = de_ranking_loss(predicted, target, reference, temperature=.1, top_k=4, block_size=3)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0


def test_rff_mmd_reproducible_and_differentiable():
    x = torch.nn.Parameter(torch.rand(10, 6)); y = torch.rand(9, 6)
    one, two = RFFMMD(6, 16, 1., 7), RFFMMD(6, 16, 1., 7)
    assert torch.equal(one.omega, two.omega)
    loss = one(x, y); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_differentiable_generation_fixed_residuals():
    centers = torch.nn.Parameter(torch.zeros(2, 3)); dispersions = torch.ones(2, 3)
    groups = torch.tensor([0, 1, 0]); residuals = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1.]])
    value = differentiable_population(centers, dispersions, groups, residuals, torch.ones(4, 3))
    value.sum().backward()
    assert value.shape == (3, 4) and torch.isfinite(centers.grad).all()


def test_relative_selection_score_and_early_stopping_resume():
    copy = {name: 2. for name in ("de_ranking", "rff_mmd", "structure", "proportion")}
    model = {name: 1. for name in copy}
    assert np.isclose(selection_score(model, copy)["score"], .5)
    stop = EarlyStopping(2); assert stop.update(.1); assert not stop.update(.05)
    restored = EarlyStopping.from_state_dict(stop.state_dict())
    assert not restored.update(.04) and restored.stopped


def test_residual_group_fallback():
    chosen, fallback = residual_indices(np.array([0, 1]), np.array([0, 0]),
        np.array([[0., 0.], [9., 9.]]), np.array([[0., 0.], [.1, .1]]), 42)
    assert chosen.shape == (2,) and fallback == 1


def test_multicut_signed_de_loss_rewards_correct_up_and_down_rankings():
    target = torch.tensor([3., 2., 1., 0., -1., -2., -3.])
    predicted = target.clone().requires_grad_()
    correct = multicut_signed_de_loss(predicted, target, [1, 2, 3], temperature=.5)
    reversed_prediction = multicut_signed_de_loss(-target, target, [1, 2, 3], temperature=.5)
    assert correct < reversed_prediction
    correct.backward()
    assert torch.isfinite(predicted.grad).all()
