import os
import re
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path):
        """Args: 
                path (string): Path to dataset
        """
        self.path = path
        self.filenames = os.listdir(self.path)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.loadtxt(os.path.join(self.path, filename)).astype(np.float32)

        m = re.match(r'Bg_([0-9.]+)_Bth_([0-9.]+)_Pe_([0-9.]+)\.txt', filename)
        if not m:
            raise ValueError(f'Could not read parameter values from file'
                    ' {os.path.join(self.path, filename)}')
        params = np.array([float(i) for i in m.groups()])
        params = self.normalize(*params).astype(np.float32)

        return data, params

    def normalize(self, Bg, Bth, Pe):
        Bg /= 2
        Pe /= 30
        return np.array([Bg, Bth, Pe])

def main():
    data_path = "../grid_data/"

    #dataset = CustomDataset(data_path)

if __name__ == '__main__':
    main()