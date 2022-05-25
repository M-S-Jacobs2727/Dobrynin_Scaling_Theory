import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scaling_torch_lib as mike
import time
import math

class ConvNeuralNet(torch.nn.Module):   
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers. 
                Shape of data progresses as follows:

                Input:          (128, 128)
                Unflatten:      ( 1, 128, 128)
                Conv2d:         ( 6, 124, 124)
                Pool:           ( 6, 62, 62)
                Conv2d:         (16, 60, 60)
                Pool:           (16, 30, 30)
                Conv2d:         (64, 28, 28)
                Pool:           (64, 14, 14)
                Flatten:        (12544,) [ = 64*14*14]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
                """
        super(ConvNeuralNet, self).__init__()

        fc0 = 4*get_final_len((128,128),5, 3, 2, 2)

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 128)),
            torch.nn.Conv2d(1, 4, 5), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 4, 3), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(fc0, 256), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

def get_final_len(res, k1, k2, p1, p2):
    """Compute final output size of two sets of (conv3d, maxpool3d) layers
    using conv kernel_size and pool kernel_size of each layer.
    """

    res2 = (math.floor(((r - k1 + 1) - p1) / p1 + 1) for r in res)
    res3 = (math.floor(((r - k2 + 1) - p2) / p2 + 1) for r in res2)
    final_len = 1
    for r in res3:
        final_len *= r
    return final_len


def train( 
        model, loss_fn, optimizer, device,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0
    avg_error = 0

    for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss
        avg_error += torch.mean(torch.abs(y - pred) / y, 0)
    avg_loss/=num_batches
    avg_error/=num_batches
    
    return avg_loss, avg_error

def test( 
        model, loss_fn, device,
        num_samples, batch_size, resolution):
    model.eval()
    avg_loss = 0
    avg_error = 0
    num_batches = num_samples // batch_size
    with torch.no_grad():
        for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            #avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
    
    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error

def main(): 

    batch_size = 100

    train_size = 700000
    test_size = 300000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'{device = }')

    loss_fn = torch.nn.MSELoss()

    val = 0

    for m in range(3):

        model = ConvNeuralNet().to(device)

        print(f'Epoch\tbatch_size\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime')

        for i in range(100):

            t_start = time.perf_counter()

            optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )
            train_loss, train_error = train( 
                model, loss_fn, optimizer, device,
                train_size, batch_size, resolution=(128, 128)
            )

            test_loss, test_error = test(
                model, loss_fn, device,
                test_size, batch_size, resolution=(128, 128)
            )

            elapsed = time.perf_counter() - t_start
            print(f'{i+1}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}')

            if (np.sum(train_error.cpu().detach().numpy()<0.1)==3) and (np.sum(test_error.cpu().detach().numpy()<0.1)==3):
                torch.save(model.state_dict(), "model.pt")
                print("Model saved!")
                break

if __name__ == '__main__':

    main()
