import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
import test_network_serial as test_gen 
import collections

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

    for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bth(num_batches, batch_size, device, return_nw=True, resolution=resolution)):

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
        for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bth(num_batches, batch_size, device, return_nw=True, resolution=resolution)):
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

def test_model(model, device, resolution, X_2, table_vals):

    model.eval()
    with torch.no_grad():
        pred = model(X_2)

    bth_pred = mike.unnormalize_Bth_param(pred)

    table_vals["Pred Bth"] = bth_pred.tolist()

    bth_error = test_gen.get_error_bth(table_vals)

    return bth_error

def main():

    batch_size = 64
    train_size = 51200
    test_size = 21952
    eval_size = test_size
    epochs = 300
    epoch_best_accuracy = 0
    resolution = (224, 224)

    path_read = "/proj/avdlab/projects/Solutions_ML/exp-data/"
    df = pd.read_csv(f"{path_read}exp-data-rcs.csv")

    epoch_best_accuracy = 0
    epoch_best_loss = 0
    epoch_best_exp = 100

    model = VGG13_Net()
 
    device = torch.device(f'cuda:1')
    model.to(device)
    print(f'{device = }')

    # init data table of all systems
    table_vals = test_gen.get_all_systems_table(df, device)
    X_131, X_2 = test_gen.load_exp_data(df, device, resolution)
    X_131 = X_131.to(device)
    X_2 = X_2.to(device)

    loss_fn = torch.nn.MSELoss()
    print(f'Epoch\ttrain_loss\tBth Train Error\ttest_loss\tBth Test Error\tBth Exp Data Error\ttime', flush=True)

    best_ave_accuracy = 100
    best_validation_loss = 100
    optimizer = torch.optim.Adam(model.parameters(),
    lr=0.00010140611176715872,
    betas=(0.5827765295538069, 0.9111444332457338),
    weight_decay=0,
    eps=3.5167735087130234e-09
    )

    init_train = np.zeros(train_size)
    init_valid = np.zeros(test_size)

    df_train_accuracy = pd.DataFrame(
            {
                "Bth-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bth-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_accuracy = pd.DataFrame(
            {
                "Bth-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bth-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
                "Nw-min": pd.Series(data=init_valid,dtype="float"),
                "Nw-max": pd.Series(data=init_valid,dtype="float"),
                "Num-Nw": pd.Series(data=init_valid,dtype="int"),
             }
        )

    df_train_loss = pd.DataFrame(
            {
                "Bth-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bth-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_loss = pd.DataFrame(
            {
                "Bth-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bth-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
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

        bth_train_true = mike.unnormalize_Bth_param(train_y)
        bth_train_pred = mike.unnormalize_Bth_param(train_pred)

        test_loss, valid_pred, valid_y, valid_Nw_min, valid_Nw_max, valid_num_Nw = validate(
            model, loss_fn, device,
            test_size, batch_size, resolution
        )

        test_exp_data_error = test_model(model, device, resolution, X_2, table_vals)

        bth_valid_true = mike.unnormalize_Bth_param(valid_y)
        bth_valid_pred = mike.unnormalize_Bth_param(valid_pred)

        bth_train_error = torch.mean(torch.abs(bth_train_true-bth_train_pred)/bth_train_true)

        bth_valid_error = torch.mean(torch.abs(bth_valid_true-bth_valid_pred)/bth_valid_true)

        elapsed = time.perf_counter() - t_start

        print(f'{m}\t{train_loss:>8f}\t{bth_train_error:.4f}\t{test_loss:>8f}\t{bth_valid_error:.4f}\t{test_exp_data_error:.4f}\t{elapsed:.2f}', flush=True)

        if test_exp_data_error < epoch_best_exp:

            epoch_best_exp = test_exp_data_error
            epoch_best_exp_error = m+1
            torch.save({
            'epoch': epoch_best_exp_error,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_best_exp_error.pt")

        if (bth_train_error+bth_valid_error)/2.0 <= best_ave_accuracy:

            best_ave_accuracy = (bth_train_error+bth_valid_error)/2.0
            epoch_best_accuracy = m+1
            torch.save({
            'epoch': epoch_best_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_best_accuracy.pt")

            df_train_accuracy["Bth-Train-True"] = bth_train_true.detach().cpu().numpy()
            df_train_accuracy["Bth-Train-Pred"] = bth_train_pred.detach().cpu().numpy()
            df_train_accuracy["Nw-min"] = train_Nw_min
            df_train_accuracy["Nw-max"] = train_Nw_max
            df_train_accuracy["Num-Nw"] = train_num_Nw

            df_valid_accuracy["Bth-Valid-True"] = bth_valid_true.detach().cpu().numpy()
            df_valid_accuracy["Bth-Valid-Pred"] = bth_valid_pred.detach().cpu().numpy()
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

            df_train_loss["Bth-Train-True"] = bth_train_true.detach().cpu().numpy()
            df_train_loss["Bth-Train-Pred"] = bth_train_pred.detach().cpu().numpy()
            df_train_loss["Nw-min"] = train_Nw_min
            df_train_loss["Nw-max"] = train_Nw_max
            df_train_loss["Num-Nw"] = train_num_Nw

            df_valid_loss["Bth-Valid-True"] = bth_valid_true.detach().cpu().numpy()
            df_valid_loss["Bth-Valid-Pred"] = bth_valid_pred.detach().cpu().numpy()
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
