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

torch.cuda.empty_cache()

class InceptionV1(torch.nn.Module):

    def __init__(self, aux_logits=False, num_classes=2):

        super(InceptionV1, self).__init__()

        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)       # output 48x48x64
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)                         # output 24x24x64
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)     # output 24x24x192
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2)                         # output 12x12x192

        # Inception(in_channels, ch1x1out, ch3x3red, ch3x3out, ch5x5red, ch5x5out, pool_out)
        # Inception(in_channels, ch1x1out, ch3x3red, ch3x3out, ch3x3redb, ch3x3redb2, ch3x3bout, pool_out):
        self.inception3a = Inception(192, 64, 96, 128, 16, 24, 32, 32)              # output 12x12x256
        self.inception3b = Inception(256, 128, 128, 192, 32, 64, 96, 64)            # output 12x12x480
        self.maxpool3 = torch.nn.MaxPool2d(3, stride=2)                         # output 6x6x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 32, 48, 64)             # output 6x6x512
        self.inception4b = Inception(512, 160, 112, 224, 24, 32, 64, 64)            # output 6x6x512
        self.inception4c = Inception(512, 128, 128, 256, 24, 32, 64, 64)            # output 6x6x512
        self.inception4d = Inception(512, 112, 144, 288, 32, 48, 64, 64)            # output 6x6x528
        self.inception4e = Inception(528, 256, 160, 320, 32, 64, 128, 128)          # output 6x6x832
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2)                         # output 3x3x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 64, 128, 128)          # output 3x3x1024
        self.inception5b = Inception(832, 384, 192, 384, 48, 64, 128, 128)         # output 3x3x1024

        self.avgpool = torch.nn.AvgPool2d(kernel_size=2)                        # output 2x2x1024
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1000)
        self.fc3 = torch.nn.Linear(1000, num_classes)


        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
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
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2

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

#        self.branch3 = torch.nn.Sequential(
#                ConvBlock(in_channels, ch5x5red, kernel_size=1, padding=0),
#                ConvBlock(ch5x5red, ch5x5out, kernel_size=3, padding=1)
#                )
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


class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes):

        super(InceptionAux, self).__init__()

        self.pool = torch.nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
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

    # debug #
    # train_pred_eta_sp = torch.zeros(num_samples, 96, 96).to(device)
    ########

    for b, (X, y, eta_sp) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):

        #train_pred_eta_sp = eta_sp
        optimizer.zero_grad()
        pred = model(X)
        Bg_pred, Bth_pred = mike.unnormalize_B_params(pred) 
        #pred_unnorm = mike.unnormalize_B_params(pred)
        #Bg = pred_unnorm[:,0]
        #Bth = pred_unnorm[:,1]
        pred_new = torch.clone(pred)
        Pe = ryan.recalc_pe_prediction(device, Bg_pred, Bth_pred, eta_sp, resolution, batch_size).to(device)
        pred_new = torch.column_stack((pred_new, torch.zeros(batch_size,1).to(device)))
        pred_new[:,2] = mike.normalize_Pe(Pe.squeeze())
        # take loss of normalized prediction and true values of Bg and Bth
        loss = loss_fn(pred, y[:,0:2])
        loss.backward()
        optimizer.step()
        avg_loss += loss
        avg_error += torch.mean(torch.abs(y - pred_new) / y, 0)

        train_pred[counter:counter+batch_size] = pred_new
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
        for b, (X, y, eta_sp) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
            pred = model(X)
            Bg, Bth = mike.unnormalize_B_params(pred)
            #Bg = pred_unnorm[:,0]
            #Bth = pred_unnorm[:,1]
            pred_new = torch.clone(pred)
            Pe = ryan.recalc_pe_prediction(device, Bg, Bth, eta_sp, resolution, batch_size).to(device)
            pred_new = torch.column_stack((pred_new, torch.zeros(batch_size,1).to(device)))
            pred_new[:,2] = mike.normalize_Pe(Pe.squeeze())
            loss = loss_fn(pred, y[:,0:2])
            avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred_new) / y, 0)

            all_data_pred[counter:counter+batch_size] = pred_new
            all_data_y[counter:counter+batch_size] = y
            counter = counter + batch_size
    
    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error, all_data_pred, all_data_y

def main():

    batch_size = [64]
    train_size = 70016
    test_size = 30016
    eval_size = test_size
    epochs = 100
    epoch_best_accuracy = 0
    resolution = (96, 96)

    epoch_best_accuracy = 0
    epoch_best_loss = 0

    model = InceptionV1()
    
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f'{device = }')

    #loss_fn = loss_functions.CustomMSELoss()
    loss_fn = torch.nn.MSELoss()
    print(f'Epoch\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime', flush=True)

    for batch in batch_size:
        best_ave_accuracy = 100
        best_validation_loss = 100

        for m in range(epochs):

            t_start = time.perf_counter()

            optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )

            train_loss, train_error, train_pred, train_y = train( 
                model, loss_fn, optimizer, device,
                train_size, batch, resolution
            )
            #for n in model.modules():
            #    if isinstance(n, torch.nn.Conv2d):
            #        print(n.weight.data)
            test_loss, test_error, valid_pred, valid_y = validate(
                model, loss_fn, device,
                test_size, batch, resolution
            )

            elapsed = time.perf_counter() - t_start
            print(f'{m+1}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}', flush=True)

            if (torch.mean(train_error[0] + train_error[1]) + torch.mean(test_error[0] + test_error[1])) / 2.0 <= best_ave_accuracy:

                best_ave_accuracy = (torch.mean(train_error) + torch.mean(test_error)) / 2.0
                epoch_best_accuracy = m+1
                torch.save({
                'epoch': epoch_best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, "model_best_accuracy_chkpt_3_inputv2.pt")

            if test_loss < best_validation_loss:
                epoch_best_loss = m+1
                best_validation_loss = test_loss
                torch.save({
                'epoch': epoch_best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, "model_best_accuracy_loss_3_inputv2.pt")

        print(f'{epoch_best_accuracy = }', flush=True)
        print(f'{epoch_best_loss = }', flush=True)


if __name__ == '__main__':

    main()
