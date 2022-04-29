import re
import os
import random
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader

class NeuralNet(torch.nn.Module):
    def __init__(self, *num_per_layer):
        super(NeuralNet, self).__init__()
        self.flatten = torch.nn.Flatten()
        seq = []
        for l1, l2 in zip(num_per_layer[:-1], num_per_layer[1:]):
            seq.append(torch.nn.Linear(l1, l2))
            seq.append(torch.nn.ReLU())
        seq.append(torch.nn.Linear(num_per_layer[-1], 3))
        self.linear_relu_stack = torch.nn.Sequential(*seq)
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CustomDataset(Dataset):
    """Pixelated viscosity plots dataset. Each datum is a 32x32 grid of values
    0 <= x <= 1. 
    """
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

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 4 == 0:
            loss, current = loss.item(), (batch_num + 1) * len(X)
            print(f'{loss:.7f} [{current:.5d}/{size:.5d}]')

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, avg_error = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            avg_error += abs(pred - y) / y
        
    avg_loss /= num_batches
    avg_error /= num_batches
    avg_error = torch.mean(avg_error, 0)
    print(f'{avg_error[0]:.3f}, {avg_error[1]:.3f}, {avg_error[2]:.3f}, {avg_loss:.3f}')

def main():
    random.seed(5)
    batch_size = 64
    train_size = 8192
    test_size = 256
    
    print('Loading dataset...')
    my_dataset = CustomDataset('../grid_data')
    print('Done.')
    first_layer_shape = my_dataset[0][0].shape
    first_layer_len = first_layer_shape[0] * first_layer_shape[1]
    num_per_layer = [first_layer_len, 64, 64]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = NeuralNet(*num_per_layer).to(device)
    print('Loaded model.')

    num_samples = len(my_dataset)
    samples = random.sample(list(np.arange(num_samples)), train_size+test_size)

    train_data = []
    train_samples = samples[:train_size]
    for s in train_samples:
        train_data.append(my_dataset[s])
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    print('Set training data.')

    test_data = []
    test_samples = samples[train_size:]
    for s in test_samples:
        test_data.append(my_dataset[s])
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print('Set testing data.')

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print('Training...')
    train(train_dataloader, model, loss_fn, optimizer, device)
    print('Done.\nTesting...')
    test(test_dataloader, model, loss_fn, device)
    print('Done.')

if __name__ == '__main__':
    main()
