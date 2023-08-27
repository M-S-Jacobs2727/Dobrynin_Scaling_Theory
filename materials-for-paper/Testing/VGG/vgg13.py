import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
#import save_model_data as savechkpt
import collections
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple

pd.options.mode.chained_assignment = None  #

torch.cuda.empty_cache()

dtype=torch.float

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(1, 1e6)

BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

NUM_BIN = 224

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

    for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):

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
        for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator(num_batches, batch_size, device, resolution=resolution)):
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
