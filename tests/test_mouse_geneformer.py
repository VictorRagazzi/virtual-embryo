import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from src.approaches.mouse_geneformer.data import (
    prepare_expression, select_future_neighbors, split_cells, tokenize_expression,
)
from src.approaches.mouse_geneformer.model import TemporalGeneformer
from src.approaches.mouse_geneformer.temporal import build_pairs, predict, train
from src.scripts.split_by_day import split_by_day


def test_split_preserves_expression_and_metadata(tmp_path):
    original = ad.AnnData(sparse.csr_matrix([[1, 0], [2, 3], [0, 4]], dtype=np.float32),
                         obs=pd.DataFrame({"day": [2, 1, 2], "label": ["a", "b", "c"]}, index=["x", "y", "z"]),
                         var=pd.DataFrame(index=["g1", "g2"]))
    original.layers["counts"] = original.X.copy()
    path = tmp_path / "source.h5ad"
    original.write_h5ad(path)
    manifest = split_by_day(path, tmp_path / "days", "day", celltype_column="label")
    assert sum(record["cells"] for record in manifest) == 3
    result = ad.read_h5ad(tmp_path / "days/day_2.h5ad")
    assert result.obs_names.tolist() == ["x", "z"]
    assert result.var_names.equals(original.var_names)
    np.testing.assert_array_equal(result.X.toarray(), original.X[[0, 2]].toarray())
    np.testing.assert_array_equal(result.layers["counts"].toarray(), result.X.toarray())
    assert result.obs.celltype.tolist() == ["a", "c"]
    with pytest.raises(FileExistsError):
        split_by_day(path, tmp_path / "days", "day")


def test_neighbors_allow_repeated_destination():
    indices, distances = select_future_neighbors(np.array([[0.0], [0.2], [10.0]]), np.array([[9.0], [0.0]]))
    np.testing.assert_array_equal(indices, [1, 1, 0])
    np.testing.assert_allclose(distances, [0, 0.2, 1])


def test_normalization_and_token_rank():
    counts = sparse.csr_matrix([[10, 4, 0]], dtype=np.float32)
    linear, expression = prepare_expression(counts, "counts")
    _, from_log = prepare_expression(sparse.csr_matrix(np.log1p(counts.toarray())), "log1p")
    np.testing.assert_allclose(expression.toarray(), from_log.toarray(), rtol=1e-6)
    ids, mask = tokenize_expression(linear, ["a", "b", "c"],
                                    {"<pad>": 0, "a": 1, "b": 2, "c": 3},
                                    {"a": 10, "b": 1, "c": 1}, 3)
    np.testing.assert_array_equal(ids, [[2, 1, 0]])
    np.testing.assert_array_equal(mask, [[True, True, False]])
    with pytest.raises(ValueError, match="inteiras"):
        prepare_expression(sparse.csr_matrix([[1.5, 2]]), "counts")
    with pytest.raises(ValueError, match="vazias"):
        prepare_expression(sparse.csr_matrix([[0, 0]]), "counts")


def test_groups_and_multistage_pairs_do_not_leak():
    obs = pd.DataFrame({"embryo": ["a", "a", "b", "b", "c", "c"]}, index=list("abcdef"))
    training, validation = split_cells(obs, 0.3, 42, "embryo")
    assert set(obs.iloc[training].embryo).isdisjoint(obs.iloc[validation].embryo)
    stages = {day: {"obs": obs, "expression": sparse.csr_matrix(np.random.default_rng(int(day * 10)).random((6, 4)), dtype=np.float32)} for day in [6.5, 7.5, 8.5]}
    splits = {day: (training, validation) for day in stages}
    pairs, records = build_pairs(stages, splits, 2, 42)
    assert set(zip(records.time, records.future_time)) == {(6.5, 7.5), (6.5, 8.5), (7.5, 8.5)}
    for split, rows in enumerate(pairs):
        for start, end, source, target in rows:
            assert source in splits[start][split]
            assert target in splits[end][split]
    training_cells = set(records[records.split == "train"].source_cell) | set(records[records.split == "train"].target_cell)
    validation_cells = set(records[records.split == "validation"].source_cell) | set(records[records.split == "validation"].target_cell)
    assert training_cells.isdisjoint(validation_cells)


def small_encoder():
    from transformers import BertConfig, BertModel
    return BertModel(BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=16,
                                max_position_embeddings=8, pad_token_id=0), add_pooling_layer=False)


def test_freezing_masking_and_prediction_modes():
    torch.set_num_threads(1)
    model = TemporalGeneformer(small_encoder(), 3, "delta", 1)
    assert not any(p.requires_grad for p in model.encoder.embeddings.parameters())
    assert not any(p.requires_grad for p in model.encoder.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in model.encoder.encoder.layer[1].parameters())
    model.train()
    assert not model.encoder.encoder.layer[0].training
    assert model.encoder.encoder.layer[1].training
    model.eval()
    expression = torch.ones(1, 3)
    times = torch.tensor([[8.5, 1.0]])
    short = model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), expression, times)
    padded = model(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]]), expression, times)
    torch.testing.assert_close(short, padded)
    short.sum().backward()
    assert any(p.grad is not None for p in model.encoder.encoder.layer[1].parameters())
    assert all(p.grad is None for p in model.encoder.encoder.layer[0].parameters())
    for parameter in model.head.parameters():
        parameter.data.zero_()
    torch.testing.assert_close(model(torch.tensor([[1]]), torch.tensor([[1]]), expression, times), expression)
    model.mode = "direct"
    torch.testing.assert_close(model(torch.tensor([[1]]), torch.tensor([[1]]), expression, times), torch.zeros_like(expression))


@pytest.mark.parametrize("mode,group_column", [("delta", None), ("direct", "embryo")])
def test_train_and_predict_roundtrip(tmp_path, mode, group_column):
    torch.set_num_threads(1)
    encoder = tmp_path / "encoder"
    small_encoder().save_pretrained(encoder)
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"<pad>": 0, "a": 1, "b": 2, "c": 3}))
    medians = tmp_path / "medians.json"
    medians.write_text(json.dumps({"a": 1, "b": 2, "c": 1}))
    stages = []
    for day in [6.5, 7.5, 8.5]:
        path = tmp_path / f"{day}.h5ad"
        data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(int(day * 10)).integers(1, 10, (8, 3)), dtype=np.float32),
                         obs=pd.DataFrame({"celltype": ["test"] * 8, "embryo": ["a", "b", "c", "d"] * 2}, index=[f"cell{i}" for i in range(8)]),
                         var=pd.DataFrame(index=["a", "b", "c"]))
        if day == 7.5:
            data = data[::-1].copy()
        data.write_h5ad(path)
        stages.append(f"{day}={path}")
    args = argparse.Namespace(epochs=1, batch_size=2, learning_rate=0.001, output=tmp_path / "run", seed=42,
                              tokens=tokens, medians=medians, gene_map=None, encoder=encoder, max_length=3,
                              stage=stages, expression_scale="counts", layer=None, max_cells=None,
                              validation_fraction=0.25, group_column=group_column, components=2, mode=mode,
                              trainable_layers=1, device="cpu")
    train(args)
    history = json.loads((args.output / "history.json").read_text())
    assert np.isfinite(history[0]["validation_mse"])
    if group_column:
        records = pd.read_csv(args.output / "pairs.csv")
        members = {}
        for split in ["train", "validation"]:
            rows = records[records.split == split]
            members[split] = {int(cell.removeprefix("cell")) % 4 for cell in set(rows.source_cell) | set(rows.target_cell)}
        assert members["train"].isdisjoint(members["validation"])
    destination = tmp_path / "prediction.h5ad"
    predict(argparse.Namespace(checkpoint=args.output / "best.pt", input=tmp_path / "8.5.h5ad",
                               output=destination, time=8.5, delta_time=1.0, device="cpu", batch_size=3))
    result = ad.read_h5ad(destination)
    assert result.shape == (8, 3)
    assert np.isfinite(result.X).all() and (result.X >= 0).all()
    assert (result.obs.timepoint == 9.5).all()
    assert "celltype" not in result.obs and "source_celltype" in result.obs
    assert result.var_names.tolist() == ["a", "b", "c"]
