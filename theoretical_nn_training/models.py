"""Three flexible classes for neural networks."""
from typing import Tuple

import torch


class LinearNeuralNet(torch.nn.Module):
    """The classic, fully connected neural network. Configure the hidden layer sizes
    and input resolution. Returns a torch.nn.Module for training and optimizing.

    Input:
        `resolution` (`data_processing.Resolution`) : The 2D or 3D resolution of the
            generated surfaces.
        `layer_sizes` (`tuple` of `int`s) : Number of nodes in each hidden layer of
            the returned neural network model.
    """

    def __init__(self, layer_sizes: Tuple[int, ...]) -> None:
        super().__init__()

        self.stack = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        )
        for prev, next in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Linear(prev, next))

        # logger.debug(f"Final model structure: \n{self.stack.state_dict()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class ConvNeuralNet2D(torch.nn.Module):
    """A convolutional neural network implementation for 2D images. The
    convolutional layers are structured as (convolution, ReLU activation function,
    max-pooling), the number and size of which are specified by `channels`,
    `kernel_sizes`, and `pool_sizes`. The resulting data is then flattened and sent
    through linear layers as specified in `layer_sizes`.

    Input:
        `resolution` (`data_processing.Resolution`) : The 2D resolution of the
            generated surfaces.
        `channels` (`tuple` of `int`s) : Number of convolutions applied in each
            convolutional layer.
        `kernel_sizes` (`tuple` of `int`s) : Size of the square kernel for each set
            of convolutions.
        `pool_sizes` (`tuple` of `int`s) : Size of the square kernal for each
            max-pooling.
        `layer_sizes` (`tuple` of `int`s) : Number of nodes in each hidden layer of
            the returned neural network model.

    Note: The tuples `channels`, `kernel_sizes`, and `pool_sizes` should all be of
    equal length.
    """

    def __init__(
        self,
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        pool_sizes: Tuple[int, ...],
        layer_sizes: Tuple[int, ...],
    ) -> None:

        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernel_sizes) = }, and"
                f" {len(pool_sizes) = }."
            )

        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, -1)))

        for prev_num_channels, next_num_channels, kernel_size, pool_size in zip(
            [1, *channels[:-1]], channels, kernel_sizes, pool_sizes
        ):
            self.stack.append(
                torch.nn.Conv2d(prev_num_channels, next_num_channels, kernel_size)
            )
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool2d(pool_size))

        self.stack.append(torch.nn.Flatten())
        self.stack.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        for prev, next in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Linear(prev, next))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class ConvNeuralNet3D(torch.nn.Module):
    """A convolutional neural network implementation for 3D images. The
    convolutional layers are structured as (convolution, ReLU activation function,
    max-pooling), the number and size of which are specified by `channels`,
    `kernel_sizes`, and `pool_sizes`. The resulting data is then flattened and sent
    through linear layers as specified in `layer_sizes`.

    Input:
        `resolution` (`data_processing.Resolution`) : The 3D resolution of the
            generated surfaces.
        `channels` (`tuple` of `int`s) : Number of convolutions applied in each
            convolutional layer.
        `kernel_sizes` (`tuple` of `int`s) : Size of the square kernel for each set
            of convolutions.
        `pool_sizes` (`tuple` of `int`s) : Size of the square kernal for each
            max-pooling.
        `layer_sizes` (`tuple` of `int`s) : Number of nodes in each hidden layer of
            the returned neural network model.

    Note: The tuples `channels`, `kernel_sizes`, and `pool_sizes` should all be of
    equal length.
    """

    def __init__(
        self,
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        pool_sizes: Tuple[int, ...],
        layer_sizes: Tuple[int, ...],
    ) -> None:

        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernel_sizes) = }, and"
                f" {len(pool_sizes) = }."
            )

        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, -1)))

        for prev_num_channels, next_num_channels, kernel_size, pool_size in zip(
            [1, *channels[:-1]], channels, kernel_sizes, pool_sizes
        ):
            self.stack.append(
                torch.nn.Conv3d(prev_num_channels, next_num_channels, kernel_size)
            )
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool3d(pool_size))

        self.stack.append(torch.nn.Flatten())
        self.stack.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        for prev, next in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Linear(prev, next))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
