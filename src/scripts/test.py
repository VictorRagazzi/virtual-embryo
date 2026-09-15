import anndata as ad, numpy as np
a = ad.read_h5ad("data/e85_e95_ot_pairs.h5ad")
print(a)
print(a.obs.head())
npz = np.load("data/e85_e95_ot_pairs_transport.npz")
print(list(npz.keys()), {k: npz[k].shape for k in npz.files})