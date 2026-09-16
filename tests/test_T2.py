import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from src.approaches.llm.T2.data import (
    prepare_expression, pair_cells, split_cells, tokenize_expression,
)
from src.approaches.llm.T2.model import TemporalGeneformer
from src.approaches.llm.T2.temporal import build_pairs, distribution_loss, grouped_batches, predict, train
from src.approaches.llm.T2.experiment import prepare, read_sample, evaluate
from src.approaches.llm.T2 import search


def test_neighbors_allow_repeated_destination():
    indices, distances = pair_cells(np.array([[0.0], [0.2], [10.0]]), np.array([[9.0], [0.0]]))
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
    stages = {day: {"obs": obs, "expression": sparse.csr_matrix(np.random.default_rng(int(day * 10)).random((6, 4)), dtype=np.float32)} for day in [6.75, 7.25, 8.5]}
    splits = {day: (training, validation) for day in stages}
    pairs, records = build_pairs(stages, splits, 2, 42)
    assert set(zip(records.time, records.future_time)) == {(6.75, 7.25), (6.75, 8.5), (7.25, 8.5)}
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


def test_freezing_masking_and_zero_interval():
    torch.set_num_threads(1)
    model = TemporalGeneformer(small_encoder(), 3, 1)
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
    model.head[-1].bias.data.fill_(0.5)
    for delta, alpha in [(0.0, 1.0), (1.0, 0.0)]:
        result = model(torch.tensor([[1]]), torch.tensor([[1]]), expression,
                       torch.tensor([[8.5, delta]]), alpha=alpha)
        assert torch.equal(result, expression)
    normal = model(torch.tensor([[1]]), torch.tensor([[1]]), expression, times, alpha=1.0)
    doubled = model(torch.tensor([[1]]), torch.tensor([[1]]), expression, times, alpha=2.0)
    torch.testing.assert_close(doubled - expression, 2 * (normal - expression))


@pytest.mark.parametrize("expression_input,mmd_weight,group_column", [("none", 0.0, None), ("projected", 0.0, "embryo"), ("none", 0.1, None)])
def test_train_and_predict_roundtrip(tmp_path, expression_input, mmd_weight, group_column):
    torch.set_num_threads(1)
    encoder = tmp_path / "encoder"
    small_encoder().save_pretrained(encoder)
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"<pad>": 0, "a": 1, "b": 2, "c": 3}))
    medians = tmp_path / "medians.json"
    medians.write_text(json.dumps({"a": 1, "b": 2, "c": 1}))
    stages = []
    for day in [6.75, 7.25, 8.5]:
        path = tmp_path / f"{day}.h5ad"
        data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(int(day * 10)).integers(1, 10, (8, 3)), dtype=np.float32),
                         obs=pd.DataFrame({"celltype": ["test"] * 8, "embryo": ["a", "b", "c", "d"] * 2}, index=[f"cell{i}" for i in range(8)]),
                         var=pd.DataFrame(index=["a", "b", "c"]))
        if day == 7.25:
            data = data[::-1].copy()
        data.write_h5ad(path)
        stages.append(f"{day}={path}")
    args = argparse.Namespace(epochs=1, batch_size=2, learning_rate=0.001, output=tmp_path / "run", seed=42,
                              tokens=tokens, medians=medians, gene_map=None, encoder=encoder, max_length=3,
                              expression_scale="counts", layer=None, max_cells=None,
                              validation_fraction=0.3 if mmd_weight else 0.25,
                              group_column=group_column, components=2,
                              expression_input=expression_input, mmd_weight=mmd_weight,
                              trainable_layers=1, device="cpu")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"stages": stages, "evaluation": {"source": {"cells": ["cell0"]}, "target": {"cells": []}}}))
    args.manifest = manifest
    train(args)
    if expression_input == "none" and mmd_weight == 0:
        (args.output / "best.pt").write_text("arquivo antigo")
        (args.output / "history.json").write_text("arquivo antigo")
        (args.output / "pairs.csv").write_text("arquivo antigo")
        (args.output / "settings.json").write_text("arquivo antigo")
        (args.output / "outro.txt").write_text("preservar")
        train(args)
        assert (args.output / "outro.txt").read_text() == "preservar"
    records = pd.read_csv(args.output / "pairs.csv")
    assert "cell0" not in set(records.source_cell) | set(records.target_cell)
    history = json.loads((args.output / "history.json").read_text())
    assert np.isfinite(history[0]["validation_mse"])
    assert np.isfinite(history[0]["validation_loss"])
    if group_column:
        records = pd.read_csv(args.output / "pairs.csv")
        members = {}
        for split in ["train", "validation"]:
            rows = records[records.split == split]
            members[split] = {int(cell.removeprefix("cell")) % 4 for cell in set(rows.source_cell) | set(rows.target_cell)}
        assert members["train"].isdisjoint(members["validation"])
    destination = tmp_path / "prediction.h5ad"
    predict(argparse.Namespace(checkpoint=args.output / "best.pt", input=tmp_path / "8.5.h5ad",
                               output=destination, time=8.5, delta_time=1.0, alpha=1.0, device="cpu", batch_size=3))
    result = ad.read_h5ad(destination)
    assert result.shape == (8, 3)
    assert np.isfinite(result.X).all() and (result.X >= 0).all()
    assert (result.obs.timepoint == 9.5).all()
    assert "celltype" not in result.obs and "source_celltype" in result.obs
    assert result.var_names.tolist() == ["a", "b", "c"]
    if expression_input == "none" and mmd_weight == 0:
        destination.write_text("arquivo antigo")
        predict(argparse.Namespace(checkpoint=args.output / "best.pt", input=tmp_path / "8.5.h5ad",
                                   output=destination, time=8.5, delta_time=1.0,
                                   alpha=1.0, device="cpu", batch_size=3))
        assert ad.read_h5ad(destination).shape == (8, 3)
    if expression_input == "projected":
        assert torch.load(args.output / "best.pt", weights_only=True)["expression_input"] == "projected"
    zero = tmp_path / "zero.h5ad"
    predict(argparse.Namespace(checkpoint=args.output / "best.pt", input=tmp_path / "8.5.h5ad",
                               output=zero, time=8.5, delta_time=1.0, alpha=0.0, device="cpu", batch_size=3))
    _, normalized = prepare_expression(ad.read_h5ad(tmp_path / "8.5.h5ad").X, "counts")
    np.testing.assert_array_equal(ad.read_h5ad(zero).X, normalized.toarray())
    reordered_input = tmp_path / "reordered.h5ad"
    ad.read_h5ad(tmp_path / "8.5.h5ad")[:, [2, 0, 1]].copy().write_h5ad(reordered_input)
    reordered_output = tmp_path / "reordered_prediction.h5ad"
    predict(argparse.Namespace(checkpoint=args.output / "best.pt", input=reordered_input,
                               output=reordered_output, time=8.5, delta_time=0.0,
                               alpha=1.0, device="cpu", batch_size=3))
    aligned = ad.read_h5ad(reordered_output)
    assert aligned.var_names.tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(aligned.X, normalized.toarray())
    old_checkpoint = tmp_path / "old.pt"
    previous = torch.load(args.output / "best.pt", weights_only=True)
    del previous["architecture_version"]
    torch.save(previous, old_checkpoint)
    with pytest.raises(ValueError, match="Checkpoint anterior incompatível"):
        predict(argparse.Namespace(checkpoint=old_checkpoint, input=reordered_input,
                                   output=tmp_path / "old_prediction.h5ad", time=8.5,
                                   delta_time=1.0, alpha=1.0, device="cpu", batch_size=3))


def test_reserve_is_reproducible_and_excluded_before_tokenization(tmp_path):
    from src.approaches.llm.T2.data import read_stage
    for name in ['E85.h5ad', 'E95.h5ad', 'E675_ex.h5ad']:
        data = ad.AnnData(sparse.csr_matrix(np.ones((12, 3), dtype=np.float32)),
                         obs=pd.DataFrame(index=[f'{name}-{i}' for i in range(12)]),
                         var=pd.DataFrame(index=['a', 'b', 'c']))
        data.write_h5ad(tmp_path / name)
    first = prepare(tmp_path, tmp_path / 'first.json', sample_size=4)
    (tmp_path / 'first.json').write_text('arquivo antigo')
    assert prepare(tmp_path, tmp_path / 'first.json', sample_size=4) == first
    assert json.loads((tmp_path / 'first.json').read_text()) == first
    second = prepare(tmp_path, tmp_path / 'second.json', sample_size=4)
    assert first == second
    assert len(first['stages']) == 3
    sample = first['evaluation']['target']
    assert set(read_sample(sample).obs_names) == set(sample['cells'])
    stage = read_stage(sample['path'], 'counts', {'<pad>': 0, 'a': 1, 'b': 2, 'c': 3},
                       {'a': 1, 'b': 1, 'c': 1}, 3, exclude_cells=sample['cells'])
    assert len(stage['obs']) == 8
    assert set(stage['obs'].index).isdisjoint(sample['cells'])
    with pytest.raises(ValueError, match='2000'):
        prepare(tmp_path, tmp_path / 'large.json', sample_size=2001)
    with pytest.raises(ValueError, match='2000'):
        read_sample({'path': 'not_opened.h5ad', 'cells': [str(i) for i in range(2001)]})


def test_times_are_inputs_and_intervals_are_irregular():
    from src.approaches.llm.T2.temporal import TemporalPairs
    stage = {'input_ids': np.array([[1, 2]]), 'mask': np.array([[True, True]]),
             'expression': sparse.csr_matrix([[1., 2.]], dtype=np.float32)}
    dataset = TemporalPairs({6.75: stage, 7.25: stage, 8.5: stage},
                            [(6.75, 7.25, 0, 0), (7.25, 8.5, 0, 0)])
    torch.testing.assert_close(dataset[0][3], torch.tensor([6.75, .5]))
    torch.testing.assert_close(dataset[1][3], torch.tensor([7.25, 1.25]))
    model = TemporalGeneformer(small_encoder(), 2, 0).eval()
    for parameter in model.head.parameters():
        parameter.data.zero_()
    # Faça cada coordenada temporal afetar uma saída diferente.
    model.head[0].weight.data[0, -2] = 1
    model.head[0].weight.data[1, -1] = 1
    model.head[2].weight.data[0, 0] = 1
    model.head[2].weight.data[1, 1] = 1
    def forward(time, delta):
        return model(torch.tensor([[1]]), torch.tensor([[True]]), torch.ones(1, 2),
                     torch.tensor([[time, delta]]))
    assert not torch.equal(forward(6.75, .5), forward(7.25, .5))
    assert not torch.equal(forward(6.75, .5), forward(6.75, 1.25))


def test_expression_projection_uses_aligned_gene_magnitudes():
    model = TemporalGeneformer(small_encoder(), 3, 0, "projected").eval()
    for parameter in model.head.parameters():
        parameter.data.zero_()
    model.expression_projection.weight.data.zero_()
    model.expression_projection.weight.data[0, 1] = 1
    model.head[0].weight.data[0, -16] = 1
    model.head[2].weight.data[0, 0] = 1
    ids, mask, times = torch.tensor([[1]]), torch.tensor([[True]]), torch.tensor([[8.5, 1.]])
    first = torch.tensor([[1., 2., 3.]])
    second = torch.tensor([[1., 4., 3.]])
    change_first = model(ids, mask, first, times) - first
    change_second = model(ids, mask, second, times) - second
    assert change_second[0, 0] > change_first[0, 0]
    assert torch.equal(change_first[0, 1:], torch.zeros(2))


def test_distribution_batches_and_real_future_population():
    pairs = [(6.75, 8.5, source, 0) for source in range(5)] + [
        (7.25, 8.5, source, 0) for source in range(4)]
    for batch in grouped_batches(pairs, 2, True, 42):
        assert len(batch) >= 2
        assert len({pairs[index][:2] for index in batch}) == 1
    prediction = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    future = torch.tensor([[0., 1.], [1., 0.]])
    loss = distribution_loss(prediction, future)
    torch.testing.assert_close(loss, torch.tensor(0.))
    distribution_loss(prediction, future + 2).backward()
    assert prediction.grad is not None


def test_evaluate_uses_only_reserved_cells_and_same_expression_scale(tmp_path, monkeypatch):
    import veckit
    model = TemporalGeneformer(small_encoder(), 3, 1)
    for parameter in model.head.parameters():
        parameter.data.zero_()
    for name in ['E85.h5ad', 'E95.h5ad']:
        data = ad.AnnData(sparse.csr_matrix(np.arange(1, 31).reshape(10, 3), dtype=np.float32),
                         obs=pd.DataFrame({'celltype': ['a', 'b'] * 5},
                                          index=[f'{name}-{i}' for i in range(10)]),
                         var=pd.DataFrame(index=['a', 'b', 'c']))
        data.write_h5ad(tmp_path / name)
    manifest = prepare(tmp_path, tmp_path / 'manifest.json', sample_size=4)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'state_dict': model.state_dict(), 'encoder_config': model.encoder.config.to_dict(),
                'genes': ['a', 'b', 'c'], 'tokens': {'<pad>': 0, 'a': 1, 'b': 2, 'c': 3},
                'medians': {'a': 1, 'b': 1, 'c': 1}, 'gene_map': None,
                'expression_scale': 'counts', 'layer': None, 'max_length': 3,
                'architecture_version': 2, 'expression_input': 'none',
                'trainable_layers': 1, 'stages': [8.5, 9.5],
                'manifest': manifest}, checkpoint)
    calls = []
    def score(**kwargs):
        prediction = ad.read_h5ad(kwargs['input'])
        target = ad.read_h5ad(kwargs['target'])
        reference = ad.read_h5ad(kwargs['reference'])
        assert kwargs['task'] == 'T1'
        assert prediction.shape == target.shape == reference.shape == (4, 3)
        assert set(target.obs_names) == set(manifest['evaluation']['target']['cells'])
        assert set(reference.obs_names) == set(manifest['evaluation']['source']['cells'])
        assert 'celltype' not in prediction.obs
        assert prediction.var_names.equals(reference.var_names)
        np.testing.assert_allclose(prediction.X, reference.X.toarray())
        calls.append(kwargs)
        return {'metrics': {'test': 1}}
    monkeypatch.setattr(veckit, 'score', score)
    evaluate(checkpoint, tmp_path / 'evaluation', batch_size=2)
    (tmp_path / 'evaluation' / 'target.h5ad').write_text('arquivo antigo')
    (tmp_path / 'evaluation' / 'extra.txt').write_text('preservar')
    evaluate(checkpoint, tmp_path / 'evaluation', batch_size=2)
    assert len(calls) == 2
    assert (tmp_path / 'evaluation' / 'extra.txt').read_text() == 'preservar'


def test_search_keeps_best_checkpoint_and_continues_after_failure(tmp_path, monkeypatch):
    output = tmp_path / 'model'
    output.mkdir()
    (output / 'best.pt').write_text('checkpoint anterior')
    attempts = []

    def fake_train(args):
        number = int(args.output.parent.name.split('_')[-1])
        attempts.append(number)
        if number == 2:
            raise RuntimeError('memória esgotada')
        args.output.mkdir(parents=True)
        (args.output / 'best.pt').write_text(f'checkpoint {number}')

    def fake_evaluate(checkpoint, evaluation_output, **kwargs):
        number = int(checkpoint.parent.parent.name.split('_')[-1])
        metrics = {
            1: {'de_score': 0.1, 'de_direction': 0.2, 'mmd_u': 0.03, 'variogram': 0.01},
            3: {'de_score': 0.1, 'de_direction': 0.2, 'mmd_u': 0.04, 'variogram': 0.01},
            4: {'de_score': 0.1, 'de_direction': 0.2, 'mmd_u': 0.03, 'variogram': 0.005},
        }[number]
        return {'metrics': {**metrics, 'energy_distance': 0.5}, 'meta': {'task': 'T1'}}

    monkeypatch.setattr(search, 'train', fake_train)
    monkeypatch.setattr(search, 'evaluate', fake_evaluate)
    args = argparse.Namespace(max_experiments=4, results_dir=tmp_path / 'search', output=output,
                              epochs=1, batch_size=2, max_cells=8, max_length=3, components=2,
                              validation_fraction=0.2, seed=42, alpha=1.0, manifest=tmp_path / 'manifest',
                              encoder=tmp_path / 'encoder', tokens=tmp_path / 'tokens',
                              medians=tmp_path / 'medians', gene_map=None, expression_scale='counts',
                              layer=None, group_column=None, device='cpu')
    best = search.run_search(args)
    records = json.loads((args.results_dir / 'results.json').read_text())
    assert attempts == [1, 2, 3, 4]
    assert [row['status'] for row in records] == ['ok', 'error', 'ok', 'ok']
    assert [row.get('new_best') for row in records] == [True, None, False, True]
    assert 'memória esgotada' in records[1]['error']
    assert records[3]['metrics']['energy_distance'] == 0.5
    assert best['iteration'] == 4
    assert (output / 'best.pt').read_text() == 'checkpoint 4'
    assert (args.results_dir / 'previous_best.pt').read_text() == 'checkpoint anterior'


def test_search_rejects_missing_selection_metric():
    with pytest.raises(ValueError, match='de_score'):
        search.selection_key({'de_score': None, 'de_direction': 0.1,
                              'mmd_u': 0.1, 'variogram': 0.1})
