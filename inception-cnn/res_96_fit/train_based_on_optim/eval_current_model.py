import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scaling_torch_lib as mike
import pandas as pd
import save_model_data as savechkpt
import loss_functions
import matplotlib.pyplot as plt
import plot_eval

torch.cuda.empty_cache()

class ConvNeuralNet(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 512x512 of type np.float32
                Two convolutional layers, three fully connected layers.
                Shape of data progresses as follows:

                Input:          (512, 512)
                Unflatten:      ( 1, 512, 512)
                Conv2d:         ( 10, 124, 124)
                Pool:           ( 10, 62, 62)
                Conv2d:         (100, 60, 60)
                Pool:           (100, 30, 30)
                Conv2d:         (200, 28, 28)
                Pool:           (200, 14, 14)
                Conv2d:         (500, 10, 10)
                Pool:           (500, 5, 5)
                Conv2d:         (750, 1, 1)
		Pool:		(750, x, x)
		Conv2d:		(1000, x, x)
		Pool:		(2000, x, x)
                Flatten:        (1000,) [ = 1000*1*1]
                FCL:            (1000,)
                FCL:            (84,)
                FCL:            (3,)
                """
        super(ConvNeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 512)),
            torch.nn.Conv2d(1, 10, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(10, 50, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(50, 100, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(100, 500, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(500, 1000, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(1000, 1500, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(1500, 2000, 5),
            torch.nn.ReLU(),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(2000,2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000,1500),
            torch.nn.ReLU(),
            torch.nn.Linear(1500,1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000,500),
            torch.nn.ReLU(),
            torch.nn.Linear(500,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,3)
        )

    def forward(self, x):
        return self.conv_stack(x)

def test(
        model, loss_fn, device,
        num_samples, batch_size, resolution):

    model.eval()
    avg_loss = 0
    avg_error = 0

    eval_data_pred = torch.zeros(num_samples, 3). to(device)
    eval_data_y = torch.zeros(num_samples, 3).to(device)
    
    counter = 0
    num_batches = num_samples // batch_size

    with torch.no_grad():
        for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
            eval_data_pred[counter:counter+batch_size] = pred
            eval_data_y[counter:counter+batch_size] = y
            counter = counter + batch_size
    
    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error, eval_data_pred, eval_data_y

def main(d, model): 

    batch_size = 50
    #train_size = 700000//4
    #test_size = 300000//4
    test_size = 500
    eval_size = test_size
    epochs = 100
    resolution = (512, 512)

    device = torch.device(f'cuda:{d}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(d)

    print(f'{device = }')

    torch.cuda.set_device(d)
    torch.distributed.init_process_group(backend='nccl', world_size=4, init_method=None, rank=d)

    model = DistributedDataParallel(model, device_ids=[d], output_device=d)

    optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001, weight_decay=0)

    loss_fn = loss_functions.CustomMSELoss()
        
    checkpoint = torch.load("model_best_accuracy_chkpt.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    eval_loss, eval_error, eval_pred, eval_y = test(model, loss_fn, device, eval_size, batch_size, resolution)

    dist.reduce(eval_loss, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(eval_error, dst=0, op=dist.ReduceOp.SUM)

    eval_loss /= 4.0
    eval_error /= 4.0

    eval_y_list = [torch.zeros((eval_size,3), dtype=torch.float, device=device) for _ in range(4)]
    dist.all_gather(eval_y_list, eval_y)
    eval_y_cat = torch.cat((eval_y_list[0], eval_y_list[1], eval_y_list[2], eval_y_list[3])).to(0)

    eval_pred_list = [torch.zeros((eval_size,3), dtype=torch.float, device=device) for _ in range(4)]
    dist.all_gather(eval_pred_list, eval_pred)
    eval_pred_cat = torch.cat((eval_pred_list[0], eval_pred_list[1], eval_pred_list[2], eval_pred_list[3])).to(0)

    if d==0:

        print(f'Epoch\teval_loss\teval_err[0]\teval_err[1]\teval_err[2]')
        print(f'{epoch}\t{eval_loss:>5f}\t{eval_error[0]:.4f}\t{eval_error[1]:.4f}\t{eval_error[2]:.4f}')

        savechkpt.save_data_eval(eval_y_cat, eval_pred_cat)

        plot_eval.plot_data(eval_y_cat, eval_pred_cat)

if __name__ == '__main__':

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Requires at least 4 GPUs to run, but got {n_gpus}"
    model = ConvNeuralNet()
    torch.multiprocessing.spawn(main, args=(model,), nprocs=4)
