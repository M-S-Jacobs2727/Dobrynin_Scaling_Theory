import numpy as np
import torch
import scaling_torch_lib as mike
import time

class NeuralNet(torch.nn.Module):   
    """The classic, fully connected neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
        
        Three fully connected layers. 
        Shape of data progresses as follows:

                Input:          (32, 32)
                Flatten:        (1024,) [ = 32*32]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super(NeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(32*32, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

class ConvNeuralNet(torch.nn.Module):   
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers. 
                Shape of data progresses as follows:

                Input:          (64, 64)
                Unflatten:      ( 1, 64, 64)
                Conv2d:         ( 6, 30, 30)
                Pool:           ( 6, 15, 15)
                Conv2d:         (16, 13, 13)
                Pool:           (16,  6,  6)
                Flatten:        (576,) [ = 16*6*6]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super(ConvNeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 64)),
            torch.nn.Conv2d(1, 6, 5), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 3), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            #torch.view(torch.size(0), torch.size(1), -1),
            #torch.permute(2, 0, 1),
            torch.nn.Linear(64*6*6, 64*6*6), 
            torch.nn.ReLU(),
            torch.nn.Linear(64*6*6, 64*6*6), 
            torch.nn.ReLU(),
            #torch.nn.Linear(64*6*6, 6000)
            torch.nn.Linear(64*6*6, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

def train( 
        model, loss_fn, optimizer, device,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0
    avg_error = 0

    for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):

        #pred = model(X)
        #loss = loss_fn(pred, y)

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

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

            #avg_loss += loss.item()
            avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
    
    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error

def main(model):
   

    batch_size = 2000
    #train_size = 112000
    #test_size = 48000

    #train_size = [3584000]
    #test_size = [1536000]

    #train_size = [14000, 28000, 56000, 112000, 224000, 448000, 896000, 1792000, 3584000]
    #test_size = [6000, 12000, 24000, 48000, 96000, 192000, 384000, 768000, 1536000]

    train_size = [448000, 896000, 1792000, 3584000]
    test_size = [192000, 384000, 768000, 1536000]

    #train_size = [14000]
    #test_size = [6000]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f'{device = }')

    loss_fn = torch.nn.MSELoss()

    #torch.cuda.set_device(d)
    #torch.distributed.init_process_group(backend='nccl', world_size=2, init_method='file:///home/sayko/Documents/test', rank=d)
    #model = DistributedDataParallel(model, device_ids=[d], output_device=d)

    for m in range(0,len(train_size)):

        print(f'Epoch\ttrain_size\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime')

        for i in range(50):
            t_start = time.perf_counter()
            optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )
            train_loss, train_error = train( 
                model, loss_fn, optimizer, device,
                train_size[m], batch_size, resolution=(64, 64)
            )

            test_loss, test_error = test(
                model, loss_fn, device,
                test_size[m], batch_size, resolution=(64, 64)
            )

            elapsed = time.perf_counter() - t_start

            print(f'{i+1}\t{train_size[m]:.2f}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}')

if __name__ == '__main__':
    model = ConvNeuralNet()
    main(model)
