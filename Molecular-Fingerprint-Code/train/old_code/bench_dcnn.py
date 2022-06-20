import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import loss_functions

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
            torch.nn.Conv2d(10, 100, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(100, 200, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(200, 500, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(500, 750, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(750, 1000, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(1000, 2000, 5),
            torch.nn.ReLU(),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(2000,2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000,1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000,3)
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

    train_pred = torch.zeros(num_samples, 3).to(device)
    train_y = torch.zeros(num_samples, 3).to(device)

    counter = 0

    for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss
        avg_error += torch.mean(torch.abs(y - pred) / y, 0)

        train_pred[counter:counter+batch_size] = pred
        train_y[counter:counter+batch_size] = y
        counter = counter + batch_size

    avg_loss/=num_batches
    avg_error/=num_batches
    
    return avg_loss, avg_error, train_pred, train_y

def validate( 
        model, loss_fn, device,
        num_samples, batch_size, resolution):
    model.eval()
    avg_loss = 0
    avg_error = 0

    all_data_pred = torch.zeros(num_samples, 3).to(device)
    all_data_y = torch.zeros(num_samples, 3).to(device)

    counter = 0

    num_batches = num_samples // batch_size

    with torch.no_grad():
        for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
            all_data_pred[counter:counter+batch_size] = pred
            all_data_y[counter:counter+batch_size] = y
            counter = counter + batch_size
    
    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error, all_data_pred, all_data_y

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

    batch_size = 100
    train_size = 700000//4
    test_size = 300000//4
    eval_size = test_size
    best_ave_accuracy = 2
    epochs = 100
    epoch_best_accuracy = 0
    resolution = (512, 512)

    device = torch.device(f'cuda:{d}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(d)

    print(f'{device = }')

    torch.cuda.set_device(d)
    torch.distributed.init_process_group(backend='nccl', world_size=4, init_method=None, rank=d)

    model = DistributedDataParallel(model, device_ids=[d], output_device=d)

    loss_fn = loss_functions.CustomMSELoss()
        
    if d==0:
        print(f'Epoch\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime')

    for m in range(epochs):

        if d==0:
            t_start = time.perf_counter()

        optimizer = torch.optim.Adam(model.parameters(),
        lr=0.001,
        weight_decay=0
        )
        train_loss, train_error, train_pred, train_y = train( 
            model, loss_fn, optimizer, device,
            train_size, batch_size, resolution
        )
        test_loss, test_error, valid_pred, valid_y = validate(
            model, loss_fn, device,
            test_size, batch_size, resolution
        )

        dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(train_error, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(test_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(test_error, dst=0, op=dist.ReduceOp.SUM)

        train_loss /= 4.0
        train_error/= 4.0
        test_loss /= 4.0
        test_error /= 4.0

        train_y_list = [torch.zeros((train_size,3), dtype=torch.float, device=device) for _ in range(4)]
        dist.all_gather(train_y_list, train_y)
        train_y_cat = torch.cat((train_y_list[0], train_y_list[1], train_y_list[2], train_y_list[3])).to(0)

        train_pred_list = [torch.zeros((train_size,3), dtype=torch.float, device=device) for _ in range(4)]
        dist.all_gather(train_pred_list, train_pred)
        train_pred_cat = torch.cat((train_pred_list[0], train_pred_list[1], train_pred_list[2], train_pred_list[3])).to(0)

        valid_y_list = [torch.zeros((test_size,3), dtype=torch.float, device=device) for _ in range(4)]
        dist.all_gather(valid_y_list, valid_y)
        valid_y_cat = torch.cat((valid_y_list[0], valid_y_list[1], valid_y_list[2], valid_y_list[3])).to(0)

        valid_pred_list = [torch.zeros((test_size,3), dtype=torch.float, device=device) for _ in range(4)]
        dist.all_gather(valid_pred_list, valid_pred)
        valid_pred_cat = torch.cat((valid_pred_list[0], valid_pred_list[1], valid_pred_list[2], valid_pred_list[3])).to(0)

        if d == 0:

            elapsed = time.perf_counter() - t_start
            print(f'{m+1}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}')

            if (torch.mean(train_error) + torch.mean(test_error)) / 2.0 <= best_ave_accuracy:

                best_ave_accuracy = (torch.mean(train_error) + torch.mean(test_error)) / 2.0
                epoch_best_accuracy = m+1
                savechkpt.save_model_and_data_chkpt(model, optimizer, m+1, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat)

        del train_y_list, train_pred_list, valid_y_list, valid_pred_list

    if d == 0:

        savechkpt.save_model_and_data_end(model, optimizer, m+1, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat)

        print(f'{epoch_best_accuracy = }')

    torch.distributed.barrier()
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

if __name__ == '__main__':

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Requires at least 4 GPUs to run, but got {n_gpus}"
    model = ConvNeuralNet()
    torch.multiprocessing.spawn(main, args=(model,), nprocs=4)
