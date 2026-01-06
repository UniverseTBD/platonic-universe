from torch.utils.data import DataLoader, Dataset
import h5py

class Localh5pyDataset(Dataset):
    def __init__(self, file_path=None, mode=None, channel_idxs=None, physical_params=None):
        self.file = h5py.File(file_path, 'r')
        self.mode = mode
        self.channel_idxs = channel_idxs
        self.physical_params = physical_params
        
    def __len__(self):
        key = list(self.file.keys())[0]
        return len(self.file[key])

    def __getitem__(self, idx):
        if self.physical_params:
            out = {f'{self.mode}_image': self.file[f'{self.mode}_images'][idx,self.channel_idxs,:,:]}
            for param in self.physical_params:
                out[param] = self.file[param][idx]
            return out
        else:
            return {f'{self.mode}_image': self.file[f'{self.mode}_images'][idx,self.channel_idxs,:,:]}