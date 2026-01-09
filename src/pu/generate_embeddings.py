from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm

from pu.pu_datasets.local_hsc_jwst import Localh5pyDataset
from pu.models import get_adapter

def generate_embeddings(file_path, mode, batch_size, model_alias, model_size, model_name, out_dir):
    if mode == 'hsc':
        channel_idxs = [0, 1, 3]
    elif mode == 'jwst':
        channel_idxs = [0, 1, 2]
            
    dataset = Localh5pyDataset(
        file_path=file_path,
        mode=mode,
        channel_idxs=channel_idxs,
        physical_params=None
    )
    
    batch_size = batch_size
    dl = DataLoader(dataset, batch_size=batch_size)
    
    adapter_cls = get_adapter(model_alias)
    adapter = adapter_cls(model_name, model_size, alias=model_alias)
    adapter.load()
    processor = adapter.get_preprocessor(mode)
    
    z = []
    with torch.no_grad():
        for B in tqdm(dl):
            z.append(adapter.embed_for_mode(B, f"{mode}_image").cpu().detach().numpy())

    z = np.vstack(z)

    np.save(out_dir + f"embeddings_{mode}_{model_alias}_{model_size}", z)

    return z
