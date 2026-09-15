import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.scripts.subset_heart import select_heart_index, subset_heart


def test_selects_explicit_heart_labels_and_preserves_data(tmp_path):
    labels = ["LV-CM", "pSHF", "JCF", "Endocardium", "Pericardium", "Proepicardium", "Endothelium", "NCC", "Myocytes", "Foregut"]
    original = ad.AnnData(sparse.csr_matrix(np.arange(30).reshape(10, 3)),
                         obs=pd.DataFrame({"celltype": labels}, index=[f"c{i}" for i in range(10)]),
                         var=pd.DataFrame(index=["g1", "g2", "g3"]))
    original.layers["counts"] = original.X.copy()
    source, output = tmp_path / "source.h5ad", tmp_path / "heart.h5ad"
    original.write_h5ad(source)
    record = subset_heart(source, output, 8.5)
    result = ad.read_h5ad(output)
    assert result.obs.celltype.tolist() == labels[:6]
    assert record["selected_cells"] == 6
    np.testing.assert_array_equal(result.X.toarray(), original.X[:6].toarray())
    np.testing.assert_array_equal(result.layers["counts"].toarray(), result.X.toarray())
    assert result.var_names.equals(original.var_names)
    assert (result.obs.embryonic_day == 8.5).all()
    assert "not_independently_verified" in result.uns["heart_selection"]["embryonic_day_source"]
    assert "embryonic_day" not in ad.read_h5ad(source).obs
    with pytest.raises(FileExistsError):
        subset_heart(source, output, 8.5)
    original.obs["embryonic_day"] = 9.5
    original.write_h5ad(source)
    with pytest.raises(ValueError, match="conflita"):
        subset_heart(source, tmp_path / "conflict.h5ad", 8.5)


def test_index_is_metadata_and_excludes_generic_myocytes(tmp_path):
    index = pd.DataFrame({"cell_type": ["Cardiomyocytes", "First heart field", "Myocytes"],
                          "dataset_family": ["example"] * 3, "timepoint": [8.25, 9.5, 10.5],
                          "h5ad_relpath": ["absent.h5ad"] * 3})
    path = tmp_path / "index.parquet"
    index.to_parquet(path)
    summary = select_heart_index(path, tmp_path / "result")
    assert summary.cells.sum() == 2
    assert set(summary.timepoint) == {8.25, 9.5}
    assert not summary.expression_available_locally.any()
    selected = pd.read_parquet(tmp_path / "result/heart_cell_index.parquet")
    assert selected.cell_type.tolist() == ["Cardiomyocytes", "First heart field"]
