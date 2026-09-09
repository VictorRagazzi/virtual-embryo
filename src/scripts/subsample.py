import scanpy as sc

e85_full = sc.read_h5ad("data/E85.h5ad")
e95_full = sc.read_h5ad("data/E95.h5ad")

sc.pp.subsample(e85_full, n_obs=min(5000, e85_full.n_obs), random_state=0)
sc.pp.subsample(e95_full, n_obs=min(5000, e95_full.n_obs), random_state=0)

e85_full.write_h5ad("data/E85_sample.h5ad")
e95_full.write_h5ad("data/E95_sample.h5ad")