import math
from typing import Optional

import torch

from theoretical_nn_training.datatypes import Resolution


def get_final_len(res: Resolution, k: int, p: int) -> Resolution:
    return Resolution(
        math.floor(((res.phi - k + 1) - p) / p + 1),
        math.floor(((res.Nw - k + 1) - p) / p + 1),
        math.floor(((res.eta_sp - k + 1) - p) / p + 1) if res.eta_sp else 0,
    )


class LinearNeuralNet(torch.nn.Module):
    """The classic, fully connected neural network.
    TODO: Make hyperparameters accessible and tune.
    """

    def __init__(
        self, res: Resolution, l1: int, l2: int, l3: Optional[int] = None
    ) -> None:
        """Input:
                np.array of size 32x32 of type np.float32

        Three fully connected layers.
        Shape of data progresses as follows:

                Input:          (32, 32)
                Flatten:        (1024,) [ = 32*32]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super().__init__()

        l0 = res.phi * res.Nw
        if res.eta_sp:
            l0 *= res.eta_sp

        if l3:
            self.conv_stack = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(l0, l1),
                torch.nn.ReLU(),
                torch.nn.Linear(l1, l2),
                torch.nn.ReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.ReLU(),
                torch.nn.Linear(l3, 3),
            )
        else:
            self.conv_stack = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(l0, l1),
                torch.nn.ReLU(),
                torch.nn.Linear(l1, l2),
                torch.nn.ReLU(),
                torch.nn.Linear(l2, 3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class ConvNeuralNet3D(torch.nn.Module):
    def __init__(
        self,
        res: Resolution,
        c1: int = 6,
        k1: int = 3,
        p1: int = 2,
        c2: int = 16,
        k2: int = 3,
        p2: int = 2,
        l1: int = 64,
        l2: int = 64,
    ) -> None:

        super().__init__()

        r1 = get_final_len(res, k1, p1)
        r2 = get_final_len(r1, k2, p2)
        l0 = c2 * r2.phi * r2.Nw
        if res.eta_sp:
            l0 *= r2.eta_sp

        self.stack = torch.nn.Sequential(
            torch.nn.Unflatten(1, (1, res[0])),
            # Convolutional Layers
            torch.nn.Conv3d(1, c1, k1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(c1, c2, k2),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Flatten(),
            # Linear Layers
            torch.nn.Linear(l0, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
