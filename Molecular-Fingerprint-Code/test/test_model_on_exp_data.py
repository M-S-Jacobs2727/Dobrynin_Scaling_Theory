import torch
#import numpy as np
from pandas import DataFrame
from get_exp_data import read_exp_data

class ConvNeuralNet128(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self):
        """Input:
                np.array of size 128x128 of type np.float32
                Two convolutional layers, three fully connected layers.
                Shape of data progresses as follows:

                Input:          (128, 128)
                Unflatten:      ( 1, 128, 128)
                Conv2d:         ( 6, 124, 124)
                Pool:           ( 6, 62, 62)
                Conv2d:         (16, 60, 60)
                Pool:           (16, 30, 30)
                Conv2d:         (120, 28, 28)
                Pool:           (120, 14, 14)
                Conv2d:         (250, 10, 10)
                Pool:           (250, 5, 5)
                Conv2d:         (1000, 1, 1)
                Flatten:        (1000,) [ = 64*14*14]
                FCL:            (1000,)
                FCL:            (84,)
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

def eval(loss_fn, model, X, y, num_samples, device):

    x = torch.reshape(X, (num_samples, 128, 128)).to(device=device, dtype=torch.float) 

    #x = torch.split(X,128)
    model.eval()
    avg_loss = 0
    avg_error = 0
    all_data_pred = torch.zeros(num_samples, 3).to(device)
    all_data_y = torch.zeros(num_samples, 3).to(device)
    counter = 0

    y = y.to(device)

    with torch.no_grad():
        pred = model(x)
        loss = loss_fn(pred, y.to(device))
        avg_loss += loss.item()
        avg_error += torch.nanmean(torch.abs(y - pred) / y, 0)
        all_data_pred = pred
        all_data_y = y
    avg_loss /= num_samples
    avg_error /= num_samples

    return avg_loss, avg_error, all_data_pred, all_data_y

def main():

    exp_fname = ("exp_data/data.csv")
    X, y, num_samples = read_exp_data(exp_fname)

    device = torch.device('cuda')

    model = ConvNeuralNet128()
    optimizer = torch.optim.Adam(model.parameters(),
            lr=0.001,
            weight_decay=0
            )
    checkpoint = torch.load("model_19_best_accuracy.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)

    loss_fn = MSECustomLoss()

    avg_loss, avg_error, all_data_pred, all_data_y = eval(loss_fn, model, X, y, num_samples, device)

    print(f'avg_loss\tavg_err[0]\tavg_err[1]\tavg_err[2]')
    print(f'{avg_loss:>5f}\t{avg_error[0]:.4f}\t{avg_error[1]:.4f}\t{avg_error[2]:.4f}')

    pred_data = all_data_pred.cpu().numpy()
    df_pred = DataFrame(pred_data)
    df_pred.to_csv("predicted_data.csv",index=False)

    y_data = all_data_y.cpu().numpy()
    df_y = DataFrame(y_data)
    df_y.to_csv("y_data.csv",index=False)

if __name__ == '__main__':
    main()
