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

DEVICE =  torch.device(f'cuda:1') 

torch.cuda.empty_cache()

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits': Optional[torch.Tensor]}

dtype=torch.float

class Inception3(torch.nn.Module):

    def __init__(self, num_classes=1, aux_logits=False, transform_input=False, inception_blocks=None, init_weights=None):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)                 # Input of 1 channel instead of 3 channels 
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3) 
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, num_classes)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.ward("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

class InceptionA(torch.nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size = 3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size = 3, stride=2)
        
    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(torch.nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7,1), padding=(3,0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size = 1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size =(1,7), padding=(0,3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size =(1,7), padding=(0,3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1,7), padding=(0,3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7,1), padding=(3,0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
        
    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size =(1,3), padding=(0,1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size =(3,1), padding=(1,0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = torch.nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)

def train(
        model, loss_fn, optimizer, DEVICE,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0

    train_pred = torch.zeros(num_samples, 1).to(DEVICE)
    train_y = torch.zeros(num_samples, 1).to(DEVICE)

    counter = 0

    for b, (X, y) in enumerate(mike.surface_generator_Bth(num_batches, batch_size, DEVICE, return_nw=False, resolution=resolution)):

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
    model = Inception3().to(DEVICE)

    # generate optimizer

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 0.9, log=False)
    beta_2 = trial.suggest_float("beta_2", 0.75, 0.999, log=False)
    eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps)

    print(f'Epoch\tBth_Error\tTime', flush=True)

    for epoch in range(20):

        t_start = time.perf_counter()

        train_loss, train_pred, train_y = train(
                model, loss_fn, optimizer, DEVICE,
                train_size, batch_size, resolution
                )

        bth_train_true = mike.unnormalize_Bth_param(train_y)
        bth_train_pred = mike.unnormalize_Bth_param(train_pred)

        bth_train_error = torch.mean(torch.abs(bth_train_true-bth_train_pred)/bth_train_true).item()

        elapsed = time.perf_counter() - t_start

        trial.report(bth_train_error, epoch)

        if trial.should_prune():

            raise optuna.exceptions.TrialPruned()

        print(f'{epoch}\t{bth_train_error:.4f}\t{elapsed:.2f}', flush=True)

    return bth_train_error

def main():

    study = optuna.create_study(direction="minimize", study_name="Bth_Hyperparam_Tune",
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
