import numpy as np
from pu.generate_embeddings import generate_embeddings
from pu.pu_datasets.local_hsc_jwst import Localh5pyDataset
from pu.metrics import wass_distance
import os
import h5py

def intramodal_experiment(file_path=None, mode=None, model_alias=None, embeddings_dir=None, n_objects=1000, k=5, params=None, verbose=False):
    model_map = {
    "convnext": (
            ["nano", "tiny", "base", "large"],
            [f"facebook/convnextv2-{s}-22k-224" for s in ["nano", "tiny", "base", "large"]],
        )
    }
    sizes, model_names = model_map[model_alias]
    embeddings = {}
    batch_size = 200
    for size, model_name in zip(sizes, model_names): 
        if verbose:
            print(f'generating embeddings for {model_alias} {size}')
        if os.path.isfile(embeddings_dir + f"embeddings_{mode}_{model_alias}_{size}.npy"):
            embeddings[size] = np.load(embeddings_dir + f"embeddings_{mode}_{model_alias}_{size}.npy")
        else:
            z = generate_embeddings(file_path, mode, batch_size, model_alias, size, model_name, embeddings_dir)
            embeddings[size] = z

    w_ds = {}
    for i in range(len(sizes)-1):
        if verbose:
            print(f'calculating distances for {sizes[i]} vs. {sizes[i+1]}')
        w_ds[sizes[i] + ' vs. ' + sizes[i+1]] = wass_distance(
            embeddings[sizes[i]][:n_objects],
            embeddings[sizes[i+1]][:n_objects],
            k,
            params
        )

    return w_ds