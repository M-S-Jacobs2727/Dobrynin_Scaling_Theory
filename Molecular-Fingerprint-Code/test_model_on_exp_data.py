import torch
import numpy as np
import os
import math

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
                Flatten:        (12544,) [ = 64*14*14]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
                """
        super(ConvNeuralNet, self).__init__()

        fc0 = 4*get_final_len((128,128),5, 3, 2, 2)

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 128)),
            torch.nn.Conv2d(1, 4, 5),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 4, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(fc0, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.conv_stack(x)

def get_final_len(res, k1, k2, p1, p2):
    """Compute final output size of two sets of (conv3d, maxpool3d) layers
    using conv kernel_size and pool kernel_size of each layer.
    """

    res2 = (math.floor(((r - k1 + 1) - p1) / p1 + 1) for r in res)
    res3 = (math.floor(((r - k2 + 1) - p2) / p2 + 1) for r in res2)
    final_len = 1
    for r in res3:
        final_len *= r
    return final_len


def eval(loss_fn, model):

    model.eval()
    test_labels = torch.load("labels_normalized.pt")
    with torch.no_grad():
        pred = model(X)
        loss = loss_fn(pred, test_labels)

def main():

    exp_surface_read_path = "exp_data_gen/"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ConvNeuralNet.to(device)
    model.load_state_dict(torch.load("model.pt"))
    #fnames = os.listdir(exp_surface_read_path)

    loss_fn = loss_fn = torch.nn.MSELoss()

    eval(loss_fn, model)