import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scaling_torch_lib_bth as mike
import time
import math
import pandas as pd

class ConvNeuralNet128(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers.
                Shape of data progresses as follows:

                Input:          (128, 128)
                Unflatten:      ( 1, 128, 128)
                Conv2d:         ( 6, 124, 124)
                Pool:           ( 6, 62, 62)
                Conv2d:         (16, 60, 60)
                Pool:           (16, 30, 30)
                Conv2d:         (64, 28, 28)
                Pool:           (64, 14, 14)
                Flatten:        (12544,) [ = 64*14*14]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
                """
        super(ConvNeuralNet128, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 128)),
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(6, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 120, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(120, 250, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(250, 1000, 5),
            torch.nn.ReLU(),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 3)
        )

    def forward(self, x):
        return self.conv_stack(x)

class ConvNeuralNet(torch.nn.Module):   
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers. 
                Shape of data progresses as follows:

                Input:          (128, 128)
                Unflatten:      ( 1, 128, 128)
                Conv2d:         ( 6, 124, 124)
                Pool:           ( 6, 62, 62)
                Conv2d:         (16, 60, 60)
                Pool:           (16, 30, 30)
                Conv2d:         (64, 28, 28)
                Pool:           (64, 14, 14)
                Flatten:        (12544,) [ = 64*14*14]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
                """
        super(ConvNeuralNet, self).__init__()

        #fc0 = 4*get_final_len((32,32),5, 3, 2, 2)

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 64)),
            torch.nn.Conv2d(1, 6, 5), 
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(6),            ## batchnorm
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(6, 16, 3), 
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(16),          ## batchnorm
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 120, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(120, 250, 5),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(120),          ## batchnorm
            #torch.nn.MaxPool(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(250, 250), 
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(120),
            torch.nn.Linear(250, 84), 
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(84),
            torch.nn.Linear(84, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

def mse_custom_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    mask = (y_true[:,0]<y_true[:,1]**0.824)



    y_ath_err = torch.mean((y_pred[mask][:,(0,2)] - y_true[mask][:,(0,2)])**2.0,dim=1)
    y_g_err = torch.mean((y_pred[~mask] - y_true[~mask])**2.0,dim=1)

    return torch.mean(torch.cat((y_ath_err, y_g_err)))

class MSECustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor
            ) -> torch.Tensor:
        return mse_custom_loss(y_pred, y_true)

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

def test( 
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

            avg_loss += loss.item()
            #avg_loss += loss
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
            all_data_pred[counter:counter+batch_size] = pred
            all_data_y[counter:counter+batch_size] = y
            counter = counter + batch_size


    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_loss, avg_error, all_data_pred, all_data_y

def main(): 

    path = "best_model.pt"
    best_ave_accuracy = 2

    batch_size = 100

    train_size = 700000
    test_size = 300000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'{device = }')

    #loss_fn = torch.nn.MSELoss()
    loss_fn = MSECustomLoss()

    val = 0

    for m in range(1):

        model = ConvNeuralNet128().to(device)

        print(f'Epoch\ttrain_loss\ttrain_err[0]\ttrain_err[1]\ttrain_err[2]\ttest_loss\ttest_err[0]\ttest_err[1]\ttest_err[2]\ttime')

        for i in range(200):

            t_start = time.perf_counter()

            optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )
            train_loss, train_error, train_pred, train_y = train( 
                model, loss_fn, optimizer, device,
                train_size, batch_size, resolution=(128, 128)
            )

            test_loss, test_error, all_data_pred, all_data_y  = test(
                model, loss_fn, device,
                test_size, batch_size, resolution=(128, 128)
            )

            elapsed = time.perf_counter() - t_start
            print(f'{i+1}\t{train_loss:>5f}\t{train_error[0]:.4f}\t{train_error[1]:.4f}\t{train_error[2]:.4f}\t{test_loss:>5f}\t{test_error[0]:.4f}\t{test_error[1]:.4f}\t{test_error[2]:.4f}\t{elapsed:.2f}')

            if (torch.mean(train_error) + torch.mean(test_error)) / 2.0 <= best_ave_accuracy:
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "model_"+str(i)+"_best_accuracy.pt")
                best_ave_accuracy = (torch.mean(train_error) + torch.mean(test_error)) / 2.0

                pred_data = all_data_pred.cpu().numpy()
                y_data = all_data_y.cpu().numpy()
                df_pred = pd.DataFrame(pred_data)
                df_y = pd.DataFrame(y_data)
                df_pred.to_csv("valid_pred_data_chkpt.csv",index=False)
                df_y.to_csv("valid_y_data_chkpt.csv",index=False)

                train_pred_data = train_pred.cpu().detach().numpy()
                train_y_data = train_y.cpu().detach().numpy()
                df_train_pred = pd.DataFrame(train_pred_data)
                df_train_y = pd.DataFrame(train_y_data)
                df_train_pred.to_csv("train_pred_data_chkpt.csv",index=False)
                df_train_y.to_csv("train_y_data_chkpt.csv",index=False)



            if (np.sum(train_error.cpu().detach().numpy()<0.1)==2) and (np.sum(test_error.cpu().detach().numpy()<0.1)==2) and (train_error[2].cpu().detach().numpy()<0.1) and (test_error[2].cpu().detach().numpy()<0.1):
                torch.save(model.state_dict(), "model_"+str(m)+"_100_batch_10_percent.pt")
                print("Model saved!")

                pred_data = all_data_pred.cpu().numpy()
                y_data = all_data_y.cpu().numpy()
                df_pred = pd.DataFrame(pred_data)
                df_y = pd.DataFrame(y_data)
                df_pred.to_csv("valid_pred_data.csv",index=False)
                df_y.to_csv("valid_y_data.csv",index=False)

                train_pred_data = train_pred.cpu().detach().numpy()
                train_y_data = train_y.cpu().detach().numpy()
                df_train_pred = pd.DataFrame(train_pred_data)
                df_train_y = pd.DataFrame(train_y_data)
                df_train_pred.to_csv("train_pred_data.csv",index=False)
                df_train_y.to_csv("train_y_data.csv",index=False)




                break

    torch.save(model.state_dict(), "model_"+str(m)+"_100_batch_10_percent_fail.pt")
    pred_data = all_data_pred.cpu().numpy()
    y_data = all_data_y.cpu().numpy()
    df_pred = pd.DataFrame(pred_data)
    df_y = pd.DataFrame(y_data)
    df_pred.to_csv("valid_pred_data_end.csv",index=False)
    df_y.to_csv("valid_y_data_end.csv",index=False)

    train_pred_data = train_pred.cpu().detach().numpy()
    train_y_data = train_y.cpu().detach().numpy()
    df_train_pred = pd.DataFrame(train_pred_data)
    df_train_y = pd.DataFrame(train_y_data)
    df_train_pred.to_csv("train_pred_data_end.csv",index=False)
    df_train_y.to_csv("train_y_data_end.csv",index=False)


if __name__ == '__main__':

    main()
