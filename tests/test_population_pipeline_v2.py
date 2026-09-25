from pathlib import Path

import numpy as np
import torch

from src.approaches.population_transformer.anchored import (
    anchored_expression, balanced_source_indices, fit_gene_programs, weighted_population_without_replacement)
from src.approaches.population_transformer.pipeline_v2 import fixed_population_draw, fixed_residual_draw, train_run
from src.approaches.population_transformer.dataset_v2 import source_time


def _example(seed=0):
    rng = np.random.default_rng(seed); k, latent, genes, cells = 3, 4, 6, 9
    proportion = np.array([.4, .3, .3], np.float32)
    target = {"proportion": proportion, "latent_mean": rng.normal(size=(k, latent)).astype(np.float32),
              "latent_dispersion": np.abs(rng.normal(size=(k, latent))).astype(np.float32)}
    groups = fixed_population_draw(proportion, cells, seed)
    return {"features": rng.normal(size=(2*k, 2+2*latent+1)).astype(np.float32),
            "groups": np.tile(np.arange(k), 2).astype(np.int64), "target": target,
            "normalized_target": {name: torch.tensor(value) for name, value in target.items()},
            "draw_groups": torch.tensor(groups), "residuals": torch.randn(cells, latent) * .1,
            "decoder_weight": torch.randn(genes, latent) * .1, "decoder_bias": torch.zeros(genes),
            "target_expression": torch.rand(cells, genes), "reference_expression": torch.rand(cells, genes)}


def test_fixed_draws_and_training_resume(tmp_path: Path):
    assert [source_time(name) for name in ("E65_ex.h5ad", "E675_ex.h5ad", "E85.h5ad")] == [6.5, 6.75, 8.5]
    assert np.array_equal(fixed_population_draw(np.array([.5, .5]), 10, 4), fixed_population_draw(np.array([.5, .5]), 10, 4))
    rounded = np.array([0.50000006, 0.30000004, 0.20000002], dtype=np.float32)
    assert len(fixed_population_draw(rounded, 256, 42)) == 256
    latent = np.arange(12, dtype=np.float32).reshape(3, 4); labels = np.array([0, 0, 1]); groups = np.array([0, 2])
    residual, fallback = fixed_residual_draw(latent, labels, groups,
        {"latent_mean": np.stack([latent[:2].mean(0), latent[2], np.zeros(4)])}, np.zeros((3, 4)), 3)
    assert residual.shape == (2, 4) and fallback == 1
    example = _example(); scores = iter([.1, .2, .15, .14])
    validation = {"score_fn": lambda model, rff: next(scores)}
    loss = {"structural_weights": {"proportion": 1., "center": 1., "dispersion": 1.},
            "ranking_genes": 6, "rff_features": 8, "rff_sigma": 1., "ranking_weight": .1,
            "mmd_weight": .1, "ranking_temperature": .1, "top_k": 2, "ranking_block_size": 3}
    training = {"learning_rate": 1e-3, "weight_decay": 1e-4, "patience": 2, "epochs": 4, "threads": 1}
    result = train_run(examples=[example], validation=validation,
        architecture={"model_dim": 8, "heads": 2, "layers": 1, "dropout": 0.}, loss_config=loss,
        training=training, output=tmp_path, seed=42)
    assert result.best_epoch == 1 and result.checkpoint.exists() and (tmp_path / "last.pt").exists()


def test_fixed_budget_retrain_has_no_fabricated_validation_score(tmp_path: Path):
    example = _example(); loss = {
        "structural_weights": {"proportion": 1., "center": 1., "dispersion": 1.},
        "ranking_genes": 6, "rff_features": 8, "rff_sigma": 1., "ranking_weight": 0.,
        "mmd_weight": 0., "ranking_temperature": .1, "top_k": 2, "ranking_block_size": 3}
    result = train_run(examples=[example], validation={},
        architecture={"model_dim": 8, "heads": 2, "layers": 1, "dropout": 0.}, loss_config=loss,
        training={"learning_rate": 1e-3, "weight_decay": 1e-4, "patience": 9, "epochs": 3, "threads": 1},
        output=tmp_path, seed=42, checkpoint_mode="last")
    assert result.best_epoch == 2 and np.isnan(result.best_score)
    assert [row["validation_score"] for row in result.history] == [None, None, None]


def test_balanced_source_indices_exhaust_pool_before_repeating():
    centers = np.zeros((1, 2), dtype=np.float32)
    selected, fallback = balanced_source_indices(np.zeros(7, dtype=np.int64), np.zeros(4, dtype=np.int64),
                                                  centers, centers, seed=42)
    counts = np.bincount(selected, minlength=4)
    assert fallback == 0 and len(np.unique(selected[:4])) == 4
    assert counts.max() - counts.min() <= 1


def test_weighted_population_uses_unique_anchors():
    source_groups = np.repeat(np.arange(3), [5, 10, 15])
    selected = weighted_population_without_replacement(np.array([.2, .3, .5]), source_groups, 20, seed=42)
    assert len(selected) == len(np.unique(selected)) == 20
    assert selected.min() >= 0 and selected.max() < len(source_groups)


def test_anchored_programs_preserve_zeros_unless_activation_is_allowed():
    changes = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.float32)
    coefficients, programs = fit_gene_programs(changes, n_programs=2, seed=42)
    base = np.array([[2, 0, 2], [0, 2, 0]], dtype=np.float32); groups = np.array([0, 1])
    prediction = anchored_expression(base, groups, coefficients, programs)
    assert prediction[0, 1] == 0 and prediction[1, 0] == 0
    activation = np.zeros((2, 3), dtype=bool); activation[1, 2] = True
    activated = anchored_expression(base, groups, coefficients, programs, activation_mask=activation)
    assert activated[1, 2] > 0
    assert np.isfinite(activated).all() and (activated >= 0).all()
