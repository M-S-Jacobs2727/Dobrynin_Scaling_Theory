import re
import os
import random
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader

class NeuralNet(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers. 
                Shape of data progresses as follows:

                Input:          (32, 32)
                Unflatten:      ( 1, 32, 32)
                Conv2d:         ( 6, 30, 30)
                Pool:           ( 6, 15, 15)
                Conv2d:         (16, 13, 13)
                Pool:           (16,  6,  6)
                Flatten:        (576,) [ = 16*6*6]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super(NeuralNet, self).__init__()

        self.forward = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 32)),
            torch.nn.Conv2d(1, 6, 3), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 3), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(16*6*6, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

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
            loss, current = loss.item(), (batch_num + 4) * len(X)
            print(f'{loss = :>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    avg_loss, avg_error = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            avg_error += torch.abs(pred - y) / y
        
    avg_loss /= num_batches
    avg_error /= num_batches
    avg_error = torch.mean(avg_error, 0)
    print(f'{avg_error[0]:>3f}, {avg_error[1]:>3f}, {avg_error[2]:>3f}, {avg_loss = :>3f}')

def main():
    """Load dataset, then train and test network over several epochs.
    TODO: Simplify data generation, record only Bg, Bth, Pe values, then
    generate data on the fly, perhaps with GPU.
    """
    random.seed(5)
    batch_size = 1000
    train_size = 50000
    test_size = 10000
    
    my_dataset = CustomDataset('../grid_data')
    print('Loaded dataset.')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = NeuralNet().to(device)
    print('Loaded model.')

    num_samples = len(my_dataset)
    for i in range(5):
        print(f'\n****Epoch #{i}****\n')
        samples = random.sample(list(np.arange(num_samples)), train_size+test_size)

        train_data = []
        train_samples = samples[:train_size]
        for s in train_samples:
            train_data.append(my_dataset[s])
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        print(f'Set training data, length {len(train_dataloader):d}.')

        test_data = []
        test_samples = samples[train_size:]
        for s in test_samples:
            test_data.append(my_dataset[s])
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        print(f'Set testing data, length {len(test_dataloader)}.')

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        print('Training...')
        train(train_dataloader, model, loss_fn, optimizer, device)
        print('Done.\nTesting...')
        test(test_dataloader, model, loss_fn, device)
        print('Done.')

if __name__ == '__main__':
    main()
