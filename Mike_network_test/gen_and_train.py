from contextlib import ExitStack
import math
from typing import Callable, Tuple
import tqdm
import torch
import sys

import scaling_torch_lib as scaling


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class NeuralNet(torch.nn.Module):
    """The classic, fully connected neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self, shape: Tuple[int]) -> None:
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
        super(NeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(shape[0]*shape[1]*shape[2], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class ConvNeuralNet2D(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self, shape: Tuple[int]) -> None:
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers.
        """
        super(ConvNeuralNet2D, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 32)),
            torch.nn.Conv2d(1, 6, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(16*6*6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class ConvNeuralNet3D(torch.nn.Module):
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self, res: Tuple[int]) -> None:
        """Input:
        """
        super(ConvNeuralNet3D, self).__init__()

        final_len = get_final_len(res)

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, res[0])),
            torch.nn.Conv3d(1, 6, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(6, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Flatten(),
            # Fully connected layers
            torch.nn.Linear(16*final_len, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


def get_final_len(
    res: 'tuple[int]', k_size: int = 3, p_size: int = 2
) -> int:
    r_2 = (math.floor(((r - k_size + 1) - p_size) / p_size + 1) for r in res)
    r_3 = (math.floor(((r - k_size + 1) - p_size) / p_size + 1) for r in r_2)
    final_len = 1
    for r in r_3:
        final_len *= r
    return final_len


def run(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_batches: int,
    batch_size: int,
    resolution: Tuple[int],
    generator: Callable[[int, int, torch.device, Tuple[int]],
                        Tuple[torch.Tensor]],
    optimizer: torch.optim.Optimizer = None,
    disable_prog_bar: bool = False
) -> None:
    if optimizer:
        model.train()
    else:
        model.eval()

    avg_loss, avg_error = 0, 0
    # Fancy way of using no_grad with testing only, if optimizer is not def
    with ExitStack() as stack:
        if not optimizer:
            stack.enter_context(torch.no_grad())

        for X, y in tqdm.tqdm(generator(
                num_batches, batch_size, device, resolution),
                desc='batches', total=num_batches, ncols=80,
                disable=disable_prog_bar):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    avg_loss /= num_batches
    avg_error /= num_batches
    print(f'avg_error = {avg_error[0]:.5f} {avg_error[1]:.5f}'
          f' {avg_error[2]:.5f}')
    print(f'{loss = :.5f}')


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[2] == 'v':
        disable_prog_bar = True
    else:
        disable_prog_bar = False

    batch_size = 100
    train_size = 500
    test_size = 100

    device = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
    print(f'{device = }')

    loss_fn = LogCoshLoss()
    # loss_fn = torch.nn.MSELoss()

    resolution = (64, 64, 64)

    model = ConvNeuralNet3D(resolution).to(device)
    print('Loaded model.')

    for i, lr in enumerate([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]):
        print(f'\n*** Learning rate {lr} ***')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for j in range(10):
            print(f'* Epoch {j+1} *')

            print('Training')
            run(model, loss_fn, device, train_size, batch_size,
                resolution, scaling.voxel_image_generator, optimizer=optimizer,
                disable_prog_bar=disable_prog_bar)

            print('Testing')
            run(model, loss_fn, device, test_size, batch_size,
                resolution, scaling.voxel_image_generator,
                disable_prog_bar=disable_prog_bar)


if __name__ == '__main__':
    main()
