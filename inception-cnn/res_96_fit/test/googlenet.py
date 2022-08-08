import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import loss_functions
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple

torch.cuda.empty_cache()

class InceptionV1(torch.nn.Module):

    def __init__(self, aux_logits=False, num_classes=3):

        super(InceptionV1, self).__init__()

        self.aux_logits = aux_logits

        #self.unflatten = torch.nn.Unflatten(1, (1, 128))

        self.conv1 = ConvBlock(2, 64, kernel_size=7, stride=2, padding=3)       # output 112x112x64
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)                         # output 56x56x64
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)     # output 56x56x192
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2)                         # output 28x28x192

        # Inception(in_channels, ch1x1out, ch3x3red, ch3x3out, ch5x5red, ch5x5out, pool_out)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)              # output 28x28x256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)            # output 28x28x480
        self.maxpool3 = torch.nn.MaxPool2d(3, stride=2)                         # output 14x14x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)             # output 14x14x512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)            # output 14x14x512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)            # output 14x14x512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)            # output 14x14x528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)          # output 14x14x832
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2)                         # output 7x7x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)          # output 7x7x1024
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)         # output 7x7x1024

        self.avgpool = torch.nn.AvgPool2d(kernel_size=6)                        # output 1x1x1024
        self.flatten = torch.nn.Flatten()
        #self.dropout = torch.nn.Dropout(p=0.4)
        self.fc = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1000)
        self.fc3 = torch.nn.Linear(1000, num_classes)


        if aux_logits:
            self.aux1 = InceptionAux(1024, num_classes)
            self.aux2 = InceptionAux(1024, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
        #x = self.unflatten(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        #x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2

        return x

class Inception(torch.nn.Module):

    def __init__(self, in_channels, ch1x1out, ch3x3red, ch3x3out, ch5x5red, ch5x5out, pool_out):

        super(Inception, self).__init__()

        self.branch1 = ConvBlock(in_channels, ch1x1out, kernel_size=1, padding=0)

        self.branch2 = torch.nn.Sequential(
                ConvBlock(in_channels, ch3x3red, kernel_size=1, padding=0),
                ConvBlock(ch3x3red, ch3x3out, kernel_size=3, padding=1)
                )

        self.branch3 = torch.nn.Sequential(
                ConvBlock(in_channels, ch5x5red, kernel_size=1, padding=0),
                ConvBlock(ch5x5red, ch5x5out, kernel_size=3, padding=1)
                )
#        self.branch3 = torch.nn.Sequential(
#                ConvBlock(in_channels, ch3x3redb, kernel_size=1, padding=0),
#                ConvBlock(ch3x3redb, ch3x3redb2, kernel_size=3, padding=1),
#                ConvBlock(ch3x3redb2, ch3x3bout, kernel_size=3, padding=1)
#                )

        self.branch4 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, pool_out, kernel_size=1, padding=0)
                )
    
    def forward(self, x):

        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes):

        super(InceptionAux, self).__init__()

        #self.dropout = torch.nn.Dropout(p=0.7)
        self.pool = torch.nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)

        return x

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
    avg_error = 0

    train_pred = torch.zeros(num_samples, 3).to(device)
    train_y = torch.zeros(num_samples, 3).to(device)

    counter = 0

    for b, (X, y) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):

        optimizer.zero_grad()
        pred = model(X)
        # how to handle one/multiple outputs of model?
        loss_main = loss_fn(pred, y)
        #loss_aux1 = loss_fn(aux1, y)
        #loss_aux2 = loss_fn(aux2, y)

#        if aux1 == aux2 == 0:
#            loss = loss_main

#        else:

#            loss_aux1 = loss_fn(aux1, y)
#            loss_aux2 = loss_fn(aux2, y)

        #loss = loss_main + loss_aux1 * 0.2 + loss_aux2 * 0.2
        loss = loss_main

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

    batch_size = 16
    train_size = 70000//4
    test_size = 30000//4
    eval_size = test_size
    best_ave_accuracy = 2
    best_validation_loss = 2
    epochs = 25
    epoch_best_accuracy = 0
    resolution = (224, 224)

    model = InceptionV1()
    
    device = torch.device(f'cuda:{d}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(d)
    print(f'{device = }')
    torch.cuda.set_device(d)
    torch.distributed.init_process_group(backend='nccl', world_size=4, init_method=None, rank=d)
    model = DistributedDataParallel(model, device_ids=[d], output_device=d)

    loss_fn = loss_functions.CustomMSELoss()
        
    if d == 0:
        print(f'Epoch\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime', flush=True)

    for m in range(epochs):

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
            print(f'{m+1}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}', flush=True)

            if (torch.mean(train_error) + torch.mean(test_error)) / 2.0 <= best_ave_accuracy:

                best_ave_accuracy = (torch.mean(train_error) + torch.mean(test_error)) / 2.0
                epoch_best_accuracy = m+1
                savechkpt.save_model_and_data_chkpt(model, optimizer, m+1, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat)

            if test_loss < best_validation_loss:
                epoch_best_loss = m+1
                best_validation_loss = test_loss
                savechkpt.save_model_and_data_loss(model, optimizer, m+1, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat)

        del train_y_list, train_pred_list, valid_y_list, valid_pred_list

        torch.distributed.barrier()

        savechkpt.save_model_and_data_end(model, optimizer, m+1, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat)

    if d==0:

        print(f'{epoch_best_accuracy = }', flush=True)
        print(f'{epoch_best_loss = }', flush=True)

    torch.distributed.barrier()

if __name__ == '__main__':

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Requires at least 4 GPUs to run, but got {n_gpus}"
    model = InceptionV1()
    torch.multiprocessing.spawn(main, args=(model,), nprocs=4)
