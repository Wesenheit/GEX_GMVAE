import anndata as an
import numpy as np

sel = 0.85

data = an.read_h5ad("GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
data_size = data.X.shape[0]
perm = np.random.permutation(range(data_size))
id_train = perm[: int(data_size * sel)]
id_test = perm[int(data_size * sel) :]
id_gex = np.flatnonzero((data.var["feature_types"] == "GEX").values)
new_data_train = an.AnnData(
    X=data.X[id_train, :][:, id_gex], obs=data.obs.iloc[id_train, :]
)
new_data_train.layers["counts"] = data.layers["counts"][id_train, :][:, id_gex]
new_data_test = an.AnnData(
    X=data.X[id_test, :][:, id_gex], obs=data.obs.iloc[id_test, :]
)
new_data_test.layers["counts"] = data.layers["counts"][id_test, :][:, id_gex]
new_data_train.write("GEX_train_data.h5ad", compression="gzip")
new_data_test.write("GEX_test_data.h5ad", compression="gzip")
print("shape for train data: {}".format(new_data_train.X.shape))
print("shape for test data: {}".format(new_data_test.X.shape))

