import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.m0_contract import validate_task1_output
from src.scripts.copy_last import build_copy_last
from src.scripts.get_dummy import build_dummy


def reference() -> ad.AnnData:
    return ad.AnnData(
        X=sparse.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32),
        obs=pd.DataFrame(index=["source_a", "source_b"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def test_copy_last_obeys_output_contract():
    source = reference()
    copied = build_copy_last(source, seed=42, target_cells=None)
    assert validate_task1_output(copied, source.var_names)["max_stored"] == 3.0
    assert copied.obs.empty
    assert copied.obs_names.tolist() == ["copy_last_000000", "copy_last_000001"]


def test_dummy_obeys_output_contract():
    source = reference()
    dummy = build_dummy(source, n_cells=4)
    metrics = validate_task1_output(dummy, source.var_names)
    assert metrics["shape"] == "4x3"
    assert metrics["stored_values"] == 0


def test_contract_supports_backed_sparse_anndata(tmp_path):
    source = reference()
    path = tmp_path / "reference.h5ad"
    source.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        assert validate_task1_output(backed, source.var_names)["stored_values"] == 3
    finally:
        backed.file.close()


def test_contract_rejects_wrong_gene_order_dtype_and_values():
    source = reference()
    wrong_order = source[:, [1, 0, 2]].copy()
    with pytest.raises(ValueError, match="ordem/identidade"):
        validate_task1_output(wrong_order, source.var_names)
    wrong_dtype = source.copy()
    wrong_dtype.X = wrong_dtype.X.astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        validate_task1_output(wrong_dtype, source.var_names)
    negative = source.copy()
    negative.X = sparse.csr_matrix([[-1, 0, 0], [0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="negativos"):
        validate_task1_output(negative, source.var_names)
