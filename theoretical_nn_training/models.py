import math
from typing import List

import torch

from theoretical_nn_training.data_processing import Resolution


def get_final_len(res: Resolution, k: int, p: int) -> Resolution:
    """Determines the resulting resolution after a convolution with kernel size `k`
    and pool size `p`. No dilation or stride is assumed."""
    return Resolution(
        math.floor(((res.phi - k + 1) - p) / p + 1),
        math.floor(((res.Nw - k + 1) - p) / p + 1),
        math.floor(((res.eta_sp - k + 1) - p) / p + 1) if res.eta_sp else 0,
    )


class LinearNeuralNet(torch.nn.Module):
    """The classic, fully connected neural network.
    TODO: Make hyperparameters accessible and tune.
    """

    def __init__(self, res: Resolution, layers: List[int]) -> None:
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

        layers = [l0, *layers]

        self.stack = torch.nn.Sequential(
            torch.nn.Flatten(),
        )
        for prev, next in zip(layers[:-1], layers[1:]):
            self.stack.append(torch.nn.Linear(prev, next))
            self.stack.append(torch.nn.ReLU())
        self.stack.append(torch.nn.Linear(layers[-1], 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class ConvNeuralNet2D(torch.nn.Module):
    def __init__(
        self,
        res: Resolution,
        channels: List[int],
        kernels: List[int],
        pools: List[int],
        layers: List[int],
    ) -> None:

        super().__init__()

        if not (len(channels) == len(kernels) == len(pools)):
            raise ValueError(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernels) = }, and {len(pools) = }."
            )

        r = res
        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, res[0])))
        for c1, c2, k, p in zip([1, *channels[:-1]], channels, kernels, pools):
            self.stack.append(torch.nn.Conv2d(c1, c2, k))
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool2d(p))
            r = get_final_len(r, k, p)

        l0 = channels[-1] * r.phi * r.Nw
        layers = [l0, *layers]
        self.stack.append(torch.nn.Flatten())

        for prev, next in zip(layers[:-1], layers[1:]):
            self.stack.append(torch.nn.Linear(prev, next))
            self.stack.append(torch.nn.ReLU())
        self.stack.append(torch.nn.Linear(layers[-1], 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class ConvNeuralNet3D(torch.nn.Module):
    def __init__(
        self,
        res: Resolution,
        channels: List[int],
        kernels: List[int],
        pools: List[int],
        layers: List[int],
    ) -> None:

        super().__init__()

        if not (len(channels) == len(kernels) == len(pools)):
            raise ValueError(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernels) = }, and {len(pools) = }."
            )

        r = res
        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, res[0])))
        for c1, c2, k, p in zip([1, *channels[:-1]], channels, kernels, pools):
            self.stack.append(torch.nn.Conv3d(c1, c2, k))
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool3d(p))
            r = get_final_len(r, k, p)

        l0 = channels[-1] * r.phi * r.Nw * r.eta_sp
        layers = [l0, *layers]
        self.stack.append(torch.nn.Flatten())

        for prev, next in zip(layers[:-1], layers[1:]):
            self.stack.append(torch.nn.Linear(prev, next))
            self.stack.append(torch.nn.ReLU())
        self.stack.append(torch.nn.Linear(layers[-1], 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
