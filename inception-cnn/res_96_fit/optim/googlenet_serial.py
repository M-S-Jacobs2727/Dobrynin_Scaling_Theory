import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import loss_functions
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
import recalc_pe as ryan

from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import train
from ax import ChoiceParameter, ParameterType

torch.cuda.empty_cache()

device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype=torch.float
class InceptionV1(torch.nn.Module):

    def __init__(self, resolution):

        super(InceptionV1, self).__init__()

        self.resolution = resolution
        kernel_size = int(self.resolution/32-1)

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)       # output 48x48x64
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)                         # output 24x24x64
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)     # output 24x24x192
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2)                         # output 12x12x192

        # Inception(in_channels, ch1x1out, ch3x3red, ch3x3out, ch5x5red, ch5x5out, pool_out)
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
        self.avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size)                # output 1x1x1024
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1000)
        self.fc3 = torch.nn.Linear(1000, 2)

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

def net_train(net, parameters, device, dtype, resolution):

    resolution_tuple = (resolution, resolution)
    train_size = 51200
    batch_size = parameters.get("batch_size")
    num_batches = train_size // batch_size
    net.to(device=device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
        lr=parameters.get("lr"),
        betas=(parameters.get("beta_1"), parameters.get("beta_2")),
        eps=parameters.get("eps", 1e-8),
        weight_decay=parameters.get("weight_decay", 0)
        )

    num_epochs = 10
    print(parameters, flush=True)
    print('epoch\tloss\tBg error\tBth error\tPe fit error\ttime elapsed', flush=True)
    for epoch in range(num_epochs):

        t_start = time.perf_counter()

        avg_loss = 0
        avg_error = torch.zeros(1, 3).to(device)

        for b, (X, y, eta_sp) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution_tuple)):

            optimizer.zero_grad()
            pred = net(X)
            Bg_pred, Bth_pred = mike.unnormalize_B_params(pred)
            pred_new = torch.clone(pred)
            Pe = ryan.recalc_pe_prediction(device, Bg_pred, Bth_pred, eta_sp, resolution_tuple, batch_size).to(device)
            pred_new = torch.column_stack((pred_new, torch.zeros(batch_size,1).to(device)))
            pred_new[:,2] = mike.normalize_Pe(Pe.squeeze())
            # take loss of normalized prediction and true values of Bg and Bth
            loss = loss_fn(pred, y[:,0:2])
            loss.backward()
            optimizer.step()
            avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred_new) / y, 0)

        avg_loss/=num_batches
        avg_error/=num_batches

        elapsed = time.perf_counter() - t_start

        print(f'{epoch}\t{avg_loss}\t{avg_error[0,0]:.4f}\t{avg_error[0,1]:.4f}\t{avg_error[0,2]:.4f}\t{elapsed:.2f}', flush=True)
        avg_error_all = torch.mean(avg_error,1).item()
    return net, avg_error_all

def train_evaluate(parameterization):

    resolution = parameterization.get("resolution")
    untrained_net = InceptionV1(resolution)
    untrained_net.to(device)
    trained_net, avg_error = net_train(net = untrained_net, parameters=parameterization, device=device, dtype=dtype, resolution=resolution)
    print(avg_error)
    return avg_error

def main():

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{device = }')
    dtype = torch.float
    #resolution = ChoiceParameter(name="resolution", parameter_type=ParameterType.INT, values=[96, 128, 160, 192, 224, 256], sort_values=True, is_ordered=True)
    best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "value_type": "float", "type": "range", "bounds": [1e-6, 0.03], "log_scale": True},
                {"name": "batch_size", "value_type": "int", "type": "choice", "values": [16, 32, 64, 128, 256]},
                {"name": "weight_decay", "value_type": "float", "type": "range", "bounds": [0, 1]},
                {"name": "resolution", "value_type": "int", "type": "choice", "values": [96, 128, 160, 192, 224, 256]},
                {"name": "beta_1", "value_type": "float", "type": "range", "bounds": [0.75, 0.9]},
                {"name": "beta_2", "value_type": "float", "type": "range", "bounds": [0.75, 0.999]},
                {"name": "eps", "value_type": "float", "type": "range", "bounds": [1e-9, 1e-6]},
                ],
            evaluation_function=train_evaluate,
            objective_name='error',
            minimize=True
            )
    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

if __name__ == '__main__':

    main()
