import numpy as np
import torch

import mike_torch_lib as mike

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

def get_data(generator, processor, batch_size):
    y = np.random.random((batch_size, 3)).astype(np.float32)
    Bg, Bth, Pe = processor.unnormalize_params(*(y.T))
    eta_sp = generator.generate(Bg, Bth, Pe)
    eta_sp = processor.add_noise(eta_sp)
    eta_sp = processor.normalize_visc(eta_sp)
    X = processor.cap(eta_sp).astype(np.float32)
    #print(np.shape(y))
    return X, y

def train(generator, processor, 
        model, loss_fn, optimizer, device,
        num_samples, batch_size):
    model.train()
    num_batches = num_samples // batch_size

    avg_loss = 0
    avg_error = 0

    for b in range(num_batches):
        X, y = get_data(generator, processor, batch_size)
        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

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
    #print(f'Training Accuracy:\n\t{avg_loss = :>5f}\n\taverage errors ='f' {avg_error[0]:>5f} {avg_error[1]:>5f} {avg_error[2]:>5f}')
    #print(f'Train, {avg_loss = :>5f}\taverage errors ='f' {avg_error[0]:>5f} {avg_error[1]:>5f} {avg_error[2]:>5f}')

    return avg_loss, avg_error

def test(generator, processor, 
        model, loss_fn, device,
        num_samples, batch_size):
    model.eval()
    avg_loss = 0
    avg_error = 0
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
    #print(f'Testing Accuracy:\n\t{avg_loss = :>5f}\n\taverage errors ='
    #    f' {avg_error[0]:>5f} {avg_error[1]:>5f} {avg_error[2]:>5f}'
    #)
    #print(f'Test, {avg_loss = :>5f}\taverage errors ='f' {avg_error[0]:>5f} {avg_error[1]:>5f} {avg_error[2]:>5f}')

    return avg_loss, avg_error

def main():
    
    batch_size = 2000
    #train_size = 112000
    #test_size = 48000

    #train_size = [14000, 28000, 56000, 112000, 224000, 448000, 896000, 1792000, 3584000]
    #test_size = [6000, 12000, 24000, 48000, 96000, 192000, 384000, 768000, 1536000]

    train_size = [448000, 896000, 1792000, 3584000]
    test_size = [192000, 384000, 768000, 1536000]

    generator = mike.SurfaceGenerator('surface_bins.json')
    processor = mike.Processor(
        data_file='surface_bins.json',
        param_file='Bg_Bth_Pe_range.json'
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{device = }')

    loss_fn = torch.nn.MSELoss()

    #lambda_term = [0, 0.01, 0.04, 0.16, 0.64, 2.56, 10.24]
    
    for m in range(0,len(train_size)):

        model = ConvNeuralNet().to(device)
        print('Loaded model.')

        print(f'Epoch\ttrain_size\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]')

        for i in range(50):
            optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )
            train_loss, train_error = train(generator, processor, 
                model, loss_fn, optimizer, device,
                train_size[m], batch_size
            )

            test_loss, test_error = test(generator, processor,
                model, loss_fn, device,
                test_size[m], batch_size
            )
            print(f'{i+1}\t{train_size[m]:.2f}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}')

if __name__ == '__main__':
    main()
