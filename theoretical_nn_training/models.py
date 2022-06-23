"""Three flexible classes for neural networks."""
import logging
import math
from typing import Tuple

import torch


def _get_final_resolution(
    resolution: Tuple[int, int, int], kernel_size: int, pool_size: int
) -> Tuple[int, int, int]:
    """Determines the resulting resolution after a convolution with a given kernel size
    and pool size. No dilation or stride is assumed.
    """
    return (
        math.floor(((resolution[0] - kernel_size + 1) - pool_size) / pool_size + 1),
        math.floor(((resolution[1] - kernel_size + 1) - pool_size) / pool_size + 1),
        math.floor(((resolution[2] - kernel_size + 1) - pool_size) / pool_size + 1)
        if resolution[2]
        else 0,
    )


class LinearNeuralNet(torch.nn.Module):
    """The classic, fully connected neural network. Configure the hidden layer sizes
    and input resolution. Returns a torch.nn.Module for training and optimizing.

    Input:
        `resolution` (`data_processing.Resolution`) : The 2D or 3D resolution of the
            generated surfaces.
        `layer_sizes` (`tuple` of `int`s) : Number of nodes in each hidden layer of
            the returned neural network model.
    """

    def __init__(
        self, resolution: Tuple[int, int, int], layer_sizes: Tuple[int, ...]
    ) -> None:
        super().__init__()

        logger = logging.getLogger("__main__")

        if resolution[2]:
            logger.debug("Model using 3D representation of data.")
        else:
            logger.debug("Model using 2D representation of data.")

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
        resolution: Tuple[int, int, int],
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        pool_sizes: Tuple[int, ...],
        layer_sizes: Tuple[int, ...],
    ) -> None:

        super().__init__()

        logger = logging.getLogger("__main__")

        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            logger.exception(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernel_sizes) = }, and"
                f" {len(pool_sizes) = }."
            )
            raise

        if resolution[2]:
            logger.exception(
                "This model is for 2D images only, but received a 3D resolution:"
                f"{resolution}."
            )
            raise

        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, resolution[0])))

        logger.debug(
            "Channels, kernel sizes, pool_sizes, and resolutions after each"
            " set of convolutions:"
        )
        logger.debug(f"    1,   -,   -, {resolution}")
        for prev_num_channels, next_num_channels, kernel_size, pool_size in zip(
            [1, *channels[:-1]], channels, kernel_sizes, pool_sizes
        ):
            self.stack.append(
                torch.nn.Conv2d(prev_num_channels, next_num_channels, kernel_size)
            )
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool2d(pool_size))
            resolution = _get_final_resolution(resolution, kernel_size, pool_size)
            logger.debug(
                f"{next_num_channels:5d}, {kernel_size:3d}, {pool_size:3d},"
                f" {resolution}"
            )

        self.stack.append(torch.nn.Flatten())
        self.stack.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        for prev, next in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Linear(prev, next))

        # logger.debug(f"Final model structure: \n{self.stack.state_dict()}")

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
        resolution: Tuple[int, int, int],
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        pool_sizes: Tuple[int, ...],
        layer_sizes: Tuple[int, ...],
    ) -> None:

        super().__init__()

        logger = logging.getLogger("__main__")

        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            logger.exception(
                "This model requires an equal number of convolutions and pools, but"
                f" received {len(channels) = }, {len(kernel_sizes) = }, and"
                f" {len(pool_sizes) = }."
            )
            raise

        if not resolution[2]:
            logger.exception(
                "This model is for 3D images only, but received a 2D resolution:"
                f"{resolution}."
            )
            raise

        self.stack = torch.nn.Sequential(torch.nn.Unflatten(1, (1, resolution[0])))

        logger.debug("Channels and resolutions after each set of convolutions:")
        logger.debug(f"    1,   -,   -, {resolution}")
        for prev_num_channels, next_num_channels, kernel_size, pool_size in zip(
            [1, *channels[:-1]], channels, kernel_sizes, pool_sizes
        ):
            self.stack.append(
                torch.nn.Conv3d(prev_num_channels, next_num_channels, kernel_size)
            )
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.MaxPool3d(pool_size))
            resolution = _get_final_resolution(resolution, kernel_size, pool_size)
            logger.debug(
                f"{next_num_channels:5d}, {kernel_size:3d}, {pool_size:3d},"
                f" {resolution}"
            )

        self.stack.append(torch.nn.Flatten())
        self.stack.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        for prev, next in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Linear(prev, next))

        # logger.debug(f"Final model structure: \n{self.stack.state_dict()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
