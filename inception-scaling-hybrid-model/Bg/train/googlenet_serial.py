import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple

pd.options.mode.chained_assignment = None  #

torch.cuda.empty_cache()

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
        self.maxpool3 = torch.nn.MaxPool2d(3, stride=2)                         # output 6x6x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 32, 48, 64)             # output 6x6x512
        self.inception4b = Inception(512, 160, 112, 224, 24, 32, 64, 64)            # output 6x6x512
        self.inception4c = Inception(512, 128, 128, 256, 24, 32, 64, 64)            # output 6x6x512
        self.inception4d = Inception(512, 112, 144, 288, 32, 48, 64, 64)            # output 6x6x528
        self.inception4e = Inception(528, 256, 160, 320, 32, 64, 128, 128)          # output 6x6x832
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2)                         # output 3x3x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 64, 128, 128)          # output 3x3x1024
        self.inception5b = Inception(832, 384, 192, 384, 48, 64, 128, 128)         # output 3x3x1024

        # kernel size here must be adjusted to account for resolution
        self.avgpool = torch.nn.AvgPool2d(kernel_size=6)                        # output 2x2x1024
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

    train_Nw_min = np.zeros(num_samples)
    train_Nw_max = np.zeros(num_samples)
    train_num_Nw = np.zeros(num_samples)

    counter = 0

    for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bg(num_batches, batch_size, device, return_nw=True, resolution=resolution)):

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss

        train_pred[counter:counter+batch_size] = pred
        train_y[counter:counter+batch_size] = y

        train_Nw_min[counter:counter+batch_size] = Nw_min
        train_Nw_max[counter:counter+batch_size] = Nw_max
        train_num_Nw[counter:counter+batch_size] = Num_Nw

        counter = counter + batch_size

    avg_loss/=num_batches
    
    return avg_loss, train_pred, train_y, train_Nw_min, train_Nw_max, train_num_Nw

def validate( 
        model, loss_fn, device,
        num_samples, batch_size, resolution):
    model.eval()
    avg_loss = 0

    all_data_pred = torch.zeros(num_samples, 1).to(device)
    all_data_y = torch.zeros(num_samples, 1).to(device)

    counter = 0

    num_batches = num_samples // batch_size

    valid_Nw_min = np.zeros(num_samples)
    valid_Nw_max = np.zeros(num_samples)
    valid_num_Nw = np.zeros(num_samples)

    with torch.no_grad():
        for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bg(num_batches, batch_size, device, return_nw=True, resolution=resolution)):
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss

            all_data_pred[counter:counter+batch_size] = pred
            all_data_y[counter:counter+batch_size] = y
            valid_Nw_min[counter:counter+batch_size] = Nw_min
            valid_Nw_max[counter:counter+batch_size] = Nw_max
            valid_num_Nw[counter:counter+batch_size] = Num_Nw

            counter = counter + batch_size
    
    avg_loss /= num_batches

    return avg_loss, all_data_pred, all_data_y, valid_Nw_min, valid_Nw_max, valid_num_Nw

def main():

    batch_size = 64
    train_size = 51200
    test_size = 21952
    eval_size = test_size
    epochs = 100
    epoch_best_accuracy = 0
    resolution = (224, 224)

    epoch_best_accuracy = 0
    epoch_best_loss = 0

    model = InceptionV1()
    
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f'{device = }')

    loss_fn = torch.nn.MSELoss()
    print(f'Epoch\ttrain_loss\tBg Train Error\ttest_loss\tBg Test Error\ttime', flush=True)

    best_ave_accuracy = 100
    best_validation_loss = 100
    optimizer = torch.optim.Adam(model.parameters(),
    lr=0.00032847455024321627,
    betas=(0.9,0.8829179550373253),
    weight_decay=0,
    eps=3.354127206879641e-09
    )

    init_train = np.zeros(train_size)
    init_valid = np.zeros(test_size)

    df_train_accuracy = pd.DataFrame(
            {
                "Bg-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bg-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_accuracy = pd.DataFrame(
            {
                "Bg-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bg-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
                "Nw-min": pd.Series(data=init_valid,dtype="float"),
                "Nw-max": pd.Series(data=init_valid,dtype="float"),
                "Num-Nw": pd.Series(data=init_valid,dtype="int"),
             }
        )

    df_train_loss = pd.DataFrame(
            {
                "Bg-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bg-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_loss = pd.DataFrame(
            {
                "Bg-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bg-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
                "Nw-min": pd.Series(data=init_valid,dtype="float"),
                "Nw-max": pd.Series(data=init_valid,dtype="float"),
                "Num-Nw": pd.Series(data=init_valid,dtype="int"),
             }
        )

    for m in range(epochs):

        t_start = time.perf_counter()

        train_loss, train_pred, train_y, train_Nw_min, train_Nw_max, train_num_Nw = train( 
            model, loss_fn, optimizer, device,
            train_size, batch_size, resolution
        )

        bg_train_true = mike.unnormalize_Bg_param(train_y)
        bg_train_pred = mike.unnormalize_Bg_param(train_pred)

        test_loss, valid_pred, valid_y, valid_Nw_min, valid_Nw_max, valid_num_Nw = validate(
            model, loss_fn, device,
            test_size, batch_size, resolution
        )

        bg_valid_true = mike.unnormalize_Bg_param(valid_y)
        bg_valid_pred = mike.unnormalize_Bg_param(valid_pred)

        bg_train_error = torch.mean(torch.abs(bg_train_true-bg_train_pred)/bg_train_true)

        bg_valid_error = torch.mean(torch.abs(bg_valid_true-bg_valid_pred)/bg_valid_true)

        elapsed = time.perf_counter() - t_start

        print(f'{m}\t{train_loss:>8f}\t{bg_train_error:.4f}\t{test_loss:>8f}\t{bg_valid_error:.4f}\t{elapsed:.2f}', flush=True)

        if (bg_train_error+bg_valid_error)/2.0 <= best_ave_accuracy:

            best_ave_accuracy = (bg_train_error+bg_valid_error)/2.0
            epoch_best_accuracy = m+1
            torch.save({
            'epoch': epoch_best_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_best_accuracy.pt")

            df_train_accuracy["Bg-Train-True"] = bg_train_true.detach().cpu().numpy()
            df_train_accuracy["Bg-Train-Pred"] = bg_train_pred.detach().cpu().numpy()
            df_train_accuracy["Nw-min"] = train_Nw_min
            df_train_accuracy["Nw-max"] = train_Nw_max
            df_train_accuracy["Num-Nw"] = train_num_Nw

            df_valid_accuracy["Bg-Valid-True"] = bg_valid_true.detach().cpu().numpy()
            df_valid_accuracy["Bg-Valid-Pred"] = bg_valid_pred.detach().cpu().numpy()
            df_valid_accuracy["Nw-min"] = valid_Nw_min
            df_valid_accuracy["Nw-max"] = valid_Nw_max
            df_valid_accuracy["Num-Nw"] = valid_num_Nw

            df_train_accuracy.to_csv('df_train_accuracy.csv', index=False)
            df_valid_accuracy.to_csv('df_valid_accuracy.csv', index=False)

        if test_loss < best_validation_loss:
            epoch_best_loss = m+1
            best_validation_loss = test_loss
            torch.save({
            'epoch': epoch_best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_lowest_loss.pt")

            df_train_loss["Bg-Train-True"] = bg_train_true.detach().cpu().numpy()
            df_train_loss["Bg-Train-Pred"] = bg_train_pred.detach().cpu().numpy()
            df_train_loss["Nw-min"] = train_Nw_min
            df_train_loss["Nw-max"] = train_Nw_max
            df_train_loss["Num-Nw"] = train_num_Nw

            df_valid_loss["Bg-Valid-True"] = bg_valid_true.detach().cpu().numpy()
            df_valid_loss["Bg-Valid-Pred"] = bg_valid_pred.detach().cpu().numpy()
            df_valid_loss["Nw-min"] = valid_Nw_min
            df_valid_loss["Nw-max"] = valid_Nw_max
            df_valid_loss["Num-Nw"] = valid_num_Nw

            df_train_loss.to_csv('df_train_loss.csv', index=False)
            df_valid_loss.to_csv('df_valid_loss.csv', index=False)

    torch.save({
    'epoch': m+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, "model_end.pt")

    print(f'{epoch_best_accuracy = }', flush=True)
    print(f'{epoch_best_loss = }', flush=True)

if __name__ == '__main__':

    main()
