import anndata as an
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from train import plot

plt.rcParams["font.family"] = "Serif"


def plot_heatmap(data):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    arr = data.layers["counts"].toarray()
    print(np.mean(arr == 0))
    ax.imshow(arr.T, vmin=0, vmax=2, interpolation="nearest")
    ax.set_ylabel("Features")
    ax.set_xlabel("Observations")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=500, transparent=True)


data = an.read_h5ad("../data/GEX_train_data.h5ad")
data_test = an.read_h5ad("../data/GEX_test_data.h5ad")
plot_heatmap(data)
data_test.obsm["embed"] = data_test.layers["counts"]
plot(data_test, "umap_X.png", dim=None, title="Umap embedding of data")
