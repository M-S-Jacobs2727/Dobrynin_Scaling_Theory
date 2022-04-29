#from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
#from torchvision.transforms import ToTensor
#from torch.optim import Adam
#from torch import nn
import numpy as np
import torch
import os
import nn_arch
import load_data
import random

#init_lr = 1e-3
#batch_size = 64
#epochs = 10

#train_split = 0.75
#test_split = 1 - train_split

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#path_data = "grid_data\\"
#file_list = []
#for file in os.listdir(path_data):
#    if file.endswith(".txt"):
#        file_list.append(file)


#output = Net(file)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = torch.unsqueeze(X,1)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 4 == 0:
            loss, current = loss.item(), batch_num * len(X)
            #print(f'loss = {loss:.7f} [{current:.5d}/{size:.5d}]')
            print(f'loss = {loss} [{current}/{size}]')

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, avg_error = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = torch.unsqueeze(X,1)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            avg_error += abs(pred - y) / y
        
    avg_loss /= num_batches
    avg_error /= num_batches
    avg_error = torch.mean(avg_error,0)
    print(f'Test Error: Average Error =  {avg_error}, Average Loss ={avg_loss}')

def main():

    random.seed(5)
    batch_size = 50
    train_size = 50000
    test_size = 10000

    print('Loading dataset...')
    my_dataset = load_data.CustomDataset('../../grid_data')
    print('Done.')
    first_layer_shape = my_dataset[0][0].shape
    first_layer_len = first_layer_shape[0] * first_layer_shape[1]
    num_per_layer = [first_layer_len, 64, 64]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = nn_arch.Net().to(device)
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

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-1)

    print('Training...')
    train(train_dataloader, model, loss_fn, optimizer, device)
    print('Done.\nTesting...')
    test(test_dataloader, model, loss_fn, device)
    print('Done.')

if __name__ == '__main__':
    main()
#for i in file_list:
#    data = np.genfromtxt(path_data + i)
#    data_t = torch.as_tensor(data, dtype=torch.float32)
