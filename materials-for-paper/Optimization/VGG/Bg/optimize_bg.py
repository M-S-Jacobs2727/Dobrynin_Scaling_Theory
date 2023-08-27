import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
import torch.optim as optim
import torch.utils.data
import optuna
import torch.nn as nn
from collections import namedtuple

DEVICE =  torch.device(f'cuda:0')

torch.cuda.empty_cache()

dtype=torch.float

class VGG13_Net(torch.nn.Module):

    def __init__(self):
        super(VGG13_Net, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Conv2d(1, 64, 3, padding=1),       # 224, 224, 64
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),      # 224, 224, 64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 112, 112, 64
            torch.nn.Conv2d(64, 128, 3, padding=1),     # 112, 112, 128
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),    # 112, 112, 128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 56, 56, 128
            torch.nn.Conv2d(128, 256, 3, padding=1),    # 56, 56, 256
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),     # 56, 56, 256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 28, 28, 256
            torch.nn.Conv2d(256, 512, 3, padding=1),    # 28, 28, 512
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 28, 28, 512
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 14, 14, 512
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 14, 14, 512
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 14, 14, 512
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 7, 7, 512

            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(25088,4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096,1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1)
        )

    def forward(self, x):
        return self.conv_stack(x)

def train(
        model, loss_fn, optimizer, DEVICE,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0

    train_pred = torch.zeros(num_samples, 1).to(DEVICE)
    train_y = torch.zeros(num_samples, 1).to(DEVICE)

    counter = 0

    for b, (X, y) in enumerate(mike.surface_generator_Bg(num_batches, batch_size, DEVICE, return_nw=False, resolution=resolution)):

        optimizer.zero_grad()
        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss

        train_pred[counter:counter+batch_size] = pred
        train_y[counter:counter+batch_size] = y

        counter = counter + batch_size

    avg_loss/=num_batches

    return avg_loss, train_pred, train_y

def objective(trial):
    resolution = (224, 224)
    train_size = 51200
    batch_size = 64
    loss_fn = torch.nn.MSELoss()
    model = VGG13_Net().to(DEVICE)

    # generate optimizer

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 0.9, log=False)
    beta_2 = trial.suggest_float("beta_2", 0.75, 0.999, log=False)
    eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps)

    print(f'Epoch\tBg_Error\tTime', flush=True)

    for epoch in range(20):

        t_start = time.perf_counter()

        train_loss, train_pred, train_y = train(
                model, loss_fn, optimizer, DEVICE,
                train_size, batch_size, resolution
                )

        bg_train_true = mike.unnormalize_Bg_param(train_y)
        bg_train_pred = mike.unnormalize_Bg_param(train_pred)

        bg_train_error = torch.mean(torch.abs(bg_train_true-bg_train_pred)/bg_train_true).item()

        elapsed = time.perf_counter() - t_start

        trial.report(bg_train_error, epoch)

        if trial.should_prune():

            raise optuna.exceptions.TrialPruned()

        print(f'{epoch}\t{bg_train_error:.4f}\t{elapsed:.2f}', flush=True)

    return bg_train_error

def main():

    study = optuna.create_study(direction="minimize", study_name="Bg_Hyperparam_Tune",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=9, interval_steps=2, n_min_trials=5)
            )
    study.optimize(objective, n_trials=50)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

if __name__ == '__main__':

    main()
