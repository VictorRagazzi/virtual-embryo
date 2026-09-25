import numpy as np
import torch
from scipy import sparse

from src.approaches.llm.T1.fine_tuning.scgpt_model import ScGPTRegressor
from src.scripts.run_scgpt_roundtrip import NonnegativeLinearDecoder, add_oracle_residual, build_sequences, normalized_covariance_loss, select_input_genes, snap_to_reference_expression


def test_selects_vocab_genes_and_builds_cls_sequences():
    matrix = sparse.csr_matrix([[0, 2, 4], [1, 0, 6]], dtype=np.float32)
    vocab = {"g1": 1, "g3": 3, "<cls>": 9}
    genes = select_input_genes(matrix, ["g1", "g2", "g3"], vocab, 2)
    ids, values, mask = build_sequences(matrix, ["g1", "g2", "g3"], genes, vocab, 5)
    assert ids.shape == values.shape == mask.shape == (2, 3)
    assert (ids[:, 0] == vocab["<cls>"]).all()
    assert (values[:, 0] == -2).all()
    assert not mask.any()


def test_uppercase_vocab_match_is_explicit_and_not_orthology_mapping():
    matrix = sparse.csr_matrix([[1, 2]], dtype=np.float32)
    vocab = {"ACTB": 1, "MYC": 2, "<cls>": 9}
    genes = select_input_genes(matrix, ["Actb", "Myc"], vocab, 2)
    ids, _, _ = build_sequences(matrix, ["Actb", "Myc"], genes, vocab, 5)
    assert set(ids[0, 1:]) == {1, 2}


def test_encoder_exposes_cls_embedding_and_decoder_is_nonnegative():
    model = ScGPTRegressor(vocab_size=12, d_model=8, nhead=2, d_hid=8, nlayers=1, dropout=0).eval()
    ids = torch.tensor([[1, 2, 3]])
    values = torch.zeros((1, 3))
    mask = torch.zeros((1, 3), dtype=torch.bool)
    with torch.no_grad():
        states = model.encode(ids, values, mask)
    assert states.shape == (1, 3, 8)
    output = NonnegativeLinearDecoder(8, 3, hidden_dim=4)(states[:, 0])
    assert output.shape == (1, 3)
    assert torch.all(output >= 0)


def test_normalized_covariance_loss_is_zero_for_identical_population_and_positive_otherwise():
    target = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    assert normalized_covariance_loss(target, target).item() == 0.0
    prediction = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert normalized_covariance_loss(prediction, target).item() > 0.0


def test_snap_to_reference_expression_returns_nearest_rows():
    reference = np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]], dtype=np.float32)
    prediction = np.array([[0.1, 0.2], [6.0, 6.0]], dtype=np.float32)
    snapped, indices = snap_to_reference_expression(prediction, reference)
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(snapped, reference[indices])


def test_oracle_residual_restores_validation_target_exactly():
    prediction = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(add_oracle_residual(prediction, target), target)
