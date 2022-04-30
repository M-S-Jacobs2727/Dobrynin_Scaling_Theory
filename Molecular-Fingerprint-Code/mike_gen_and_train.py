import numpy as np
import torch

import mike_torch_lib as mike

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

        self.conv_stack = torch.nn.Sequential(
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
    
    def forward(self, x):
        return self.conv_stack(x)

def get_data(generator, processor, batch_size):
    y = np.random.random((batch_size, 3)).astype(np.float32)
    Bg, Bth, Pe = processor.unnormalize_params(*(y.T))
    eta_sp = generator.generate(Bg, Bth, Pe)
    eta_sp = processor.add_noise(eta_sp)
    eta_sp = processor.normalize_visc(eta_sp)
    X = processor.cap(eta_sp).astype(np.float32)
    return X, y

def train(generator, processor, 
        model, loss_fn, optimizer, device,
        num_samples, batch_size):
    model.train()
    num_batches = num_samples // batch_size
    for b in range(num_batches):
        X, y = get_data(generator, processor, batch_size)
        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 1 == 0:
            loss, current = loss.item(), (b + 1) * batch_size
            print(f'{loss = :>7f} [{current:>5d}/{num_samples:>5d}]')

def test(generator, processor, 
        model, loss_fn, device,
        num_samples, batch_size):
    model.eval()
    avg_loss = 0
    avg_error = torch.zeros((3,))
    num_batches = num_samples // batch_size
    with torch.no_grad():
        for b in range(num_batches):
            X, y = get_data(generator, processor, batch_size)
            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
    
    avg_loss /= num_batches
    avg_error /= num_batches

    print(f'Accuracy:\n\t{avg_loss = :>5f}\n\t{avg_error = :>5f}')

def main():
    
    batch_size = 10000
    train_size = 500000
    test_size = 100000

    generator = mike.SurfaceGenerator('Molecular-Fingerprint-Code/surface_bins.json')
    processor = mike.Processor(
        data_file='Molecular-Fingerprint-Code/surface_bins.json',
        param_file='Molecular-Fingerprint-Code/Bg_Bth_Pe_range.json'
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{device = }')

    model = NeuralNet().to(device)
    print('Loaded model.')

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for i in range(10):
        print(f'*** Epoch {i+1} ***')
        print('Training')
        train(generator, processor, 
            model, loss_fn, optimizer, device,
            train_size, batch_size
        )

        print('Testing')
        test(generator, processor,
            model, loss_fn, device,
            test_size, batch_size
        )

if __name__ == '__main__':
    main()