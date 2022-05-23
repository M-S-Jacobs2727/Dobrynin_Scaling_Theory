import numpy as np
import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch.distributed as dist
import scaling_torch_lib as mike
import time
import math

class ConvNeuralNet(torch.nn.Module):
    '''The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    '''

    def __init__(self, c1, c2, fc1, fc2, resolution, k1, k2):
        '''Input:
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
                FCL:            (12544,)
                FCL:            (12544,)
                FCL:            (3,)
        '''
        super(ConvNeuralNet, self).__init__()

        fc0 = c2*get_final_len(resolution, k1, k2, 2, 2)

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, resolution[0])),
            torch.nn.Conv2d(1, c1, k1), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(c1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(c1, c2, k2), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(c2),
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(fc0, fc1), 
            torch.nn.ReLU(),
            torch.nn.Linear(fc1, fc2), 
            torch.nn.ReLU(),
            torch.nn.Linear(fc2, 3)
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

def eval_hyperparams(config):

    device = torch.device("cuda:0")

    train_size = 700000
    test_size = 300000

    all_time_err = 100
    all_time_loss = 1


    model = ConvNeuralNet(config['c1'], config['c2'], config['fc1'], config['fc2'], config['resolution'], config['k1'], config['k2']).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
            lr = config["lr"],
            weight_decay=0
            )
    for i in range(20):

        train_loss, train_error = train(
                model, loss_fn, optimizer, device,
                train_size, batch_size=config["batch_size"], resolution=config["resolution"]
                )
        test_loss, test_error = test(
                model, loss_fn,  device,
                train_size, batch_size=config["batch_size"], resolution=config["resolution"]
                )
        test_error = torch.max(test_error).item()

        if test_error < all_time_err:
            all_time_err = test_error
        if test_loss < all_time_loss:
            all_time_loss = test_loss

        tune.report(test_error=test_error,test_loss=test_loss,all_time_err=all_time_err,all_time_loss=all_time_loss)

def main(num_samples=250, max_num_epochs=10, gpus_per_trial=1):

    ray.init(num_cpus=2, num_gpus=2)

    config = {
            "batch_size": tune.choice([500]),
            "c1" : tune.choice([4, 8, 16]),
            "c2" : tune.choice([4]),
            "fc1" : tune.sample_from(lambda _: 2**np.random.randint(6, 14)),
            "fc2" : tune.sample_from(lambda _: 2**np.random.randint(6, 9)),
            "k1" : tune.choice([3, 5]),
            "k2" : tune.choice([3]),
            "lr" : tune.choice([0.001]),
            "resolution" : tune.choice([(64, 64), (128, 128)])
    }

    scheduler = ASHAScheduler(
            grace_period=10,
            metric="test_loss",
            mode="min"
    )

    analysis = tune.run(
            eval_hyperparams,
            num_samples=250,
            config=config,
            scheduler=scheduler,
            resources_per_trial={"gpu": 1}
    ) 

    print("best config for test_loss: ", analysis.get_best_config(metric="test_loss", mode="min"))
    print("best config for test_error: ", analysis.get_best_config(metric="test_error", mode="min"))
    print("best config for all-time error: ", analysis.get_best_config(metric="test_error", mode="min", scope="all"))
    print("best config for all-time loss: ", analysis.get_best_config(metric="test_loss", mode="min", scope="all"))

if __name__ == '__main__':

    main()
