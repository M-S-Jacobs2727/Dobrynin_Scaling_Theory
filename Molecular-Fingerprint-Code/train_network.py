import re
import os
import random
import numpy as np
import pandas as pd
import json
import torch.nn

import generate_surfaces

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

class CustomDataLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.filenames = os.list(self.path)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.loadtxt(os.path.join(self.path, filename)).flatten()

        m = re.match(r'Bg_([0-9.]+)_Bth_([0-9.]+)_Pe_([0-9.]+)\.txt', filename)
        if not m:
            raise ValueError(f'Could not read parameter values from file'
                    ' {os.path.join(self.path, filename)}')
        params = np.array([float(i) for i in m.groups()])
        params = self.normalize(*params)

        return data, params

    def normalize(self, Bg, Bth, Pe):
        Bg /= 2
        Pe /= 30
        return Bg, Bth, Pe

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
            loss, current = loss.item(), batch_num * len(X)
            print(f'{loss = :>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            accuracy += 1 - np.abs(pred - y) / y
        
    avg_loss /= num_batches
    accuracy /= num_batches
    print(f'Test Error:\n  {accuracy = :>0.3f}, {avg_loss = :>8f} \n')

def get_num_start():
    with open('Molecular-Fingerprint-Code/surface_bins.json') as f:
        bin_data = json.load(f)
    return bin_data['Nw_num_bins'] * bin_data['phi_num_bins']

def main():
    num_first_layer = get_num_start()
    batch_size = 16
    train_size = 256
    test_size = 64
    num_per_layer = [num_first_layer, 64, 64]
    device = 'gpu' if torch.cuda.is_available() else "cpu"
    model = NeuralNet(*num_per_layer).to(device)
    print(model)

    my_dataloader = CustomDataLoader('Data')
    num_samples = len(my_dataloader)
    samples = random.sample(np.arange(num_samples), train_size+test_size)

    train_data = np.zeros(train_size)
    train_samples = samples[:train_size]
    for i, s in enumerate(train_samples):
        train_data[i] = my_dataloader[s]
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    test_samples = samples[train_size:]
    test_data = np.zeros(test_size)
    for i, s in enumerate(test_samples):
        test_data[i] = my_dataloader[s]
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)

if __name__ == '__main__':
    main()
