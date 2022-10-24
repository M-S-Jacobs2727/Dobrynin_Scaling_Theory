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

DEVICE =  torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()

device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype=torch.float
class InceptionV1(torch.nn.Module):

    def __init__(self):

        super(InceptionV1, self).__init__()

        self.conv1 = ConvBlock(1, 64, kernel_size=7, stride=2, padding=3)       # output 48x48x64
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)                         # output 24x24x64
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)     # output 24x24x192
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2)                         # output 12x12x192

        # Inception(in_channels, ch1x1out, ch3x3red, ch3x3out, ch3x3redb, ch3x3redb2, ch3x3bout, pool_out):
        self.inception3a = Inception(192, 64, 96, 128, 16, 24, 32, 32)              # output 12x12x256
        self.inception3b = Inception(256, 128, 128, 192, 32, 64, 96, 64)            # output 12x12x480
        self.maxpool3 = torch.nn.MaxPool2d(3, stride=2)                             # output 6x6x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 32, 48, 64)             # output 6x6x512
        self.inception4b = Inception(512, 160, 112, 224, 24, 32, 64, 64)            # output 6x6x512
        self.inception4c = Inception(512, 128, 128, 256, 24, 32, 64, 64)            # output 6x6x512
        self.inception4d = Inception(512, 112, 144, 288, 32, 48, 64, 64)            # output 6x6x528
        self.inception4e = Inception(528, 256, 160, 320, 32, 64, 128, 128)          # output 6x6x832
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2)                             # output 3x3x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 64, 128, 128)          # output 3x3x832
        self.inception5b = Inception(832, 384, 192, 384, 48, 64, 128, 128)          # output 3x3x1024

        # kernel size of AvgPool must represent option for resolution
        self.avgpool = torch.nn.AvgPool2d(kernel_size=6)                            # output 1x1x1024
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1000)
        self.fc3 = torch.nn.Linear(1000, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Inception(torch.nn.Module):

    #def __init__(self, in_channels, ch1x1out, ch3x3red, ch3x3out, ch5x5red, ch5x5out, pool_out):
    def __init__(self, in_channels, ch1x1out, ch3x3red, ch3x3out, ch3x3redb, ch3x3redb2, ch3x3bout, pool_out):
        super(Inception, self).__init__()

        self.branch1 = ConvBlock(in_channels, ch1x1out, kernel_size=1, padding=0)

        self.branch2 = torch.nn.Sequential(
                ConvBlock(in_channels, ch3x3red, kernel_size=1, padding=0),
                ConvBlock(ch3x3red, ch3x3out, kernel_size=3, padding=1)
                )

        self.branch3 = torch.nn.Sequential(
                ConvBlock(in_channels, ch3x3redb, kernel_size=1, padding=0),
                ConvBlock(ch3x3redb, ch3x3redb2, kernel_size=3, padding=1),
                ConvBlock(ch3x3redb2, ch3x3bout, kernel_size=3, padding=1)
                )

        self.branch4 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, pool_out, kernel_size=1, padding=0)
                )
    
    def forward(self, x):

        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)

def train(
        model, loss_fn, optimizer, device,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0

    train_pred = torch.zeros(num_samples, 1).to(device)
    train_y = torch.zeros(num_samples, 1).to(device)

    counter = 0

    for b, (X, y) in enumerate(mike.surface_generator_Bg(num_batches, batch_size, device, return_nw=False, resolution=resolution)):

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
    model = InceptionV1().to(DEVICE)

    # generate optimizer

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
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
                n_startup_trials=10, n_warmup_steps=10, interval_steps=2, n_min_trials=10)
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
