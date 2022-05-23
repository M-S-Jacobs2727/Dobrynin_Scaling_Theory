import math
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import scaling_torch_lib as scaling

PROJ_PATH = Path('/proj/avdlab/projects/Solutions_ML/')
LOG_PATH = Path(PROJ_PATH, 'mike_outputs/')
RAY_PATH = Path(LOG_PATH, 'ray_results/')


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(
            torch.tensor(2.0))
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
        super().__init__()

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
        super().__init__()

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

    def __init__(
        self,
        c1: int = 6, k1: int = 3, p1: int = 2,
        c2: int = 16, k2: int = 3, p2: int = 2,
        l1: int = 64, l2: int = 64,
        res: Tuple[int] = (32, 32, 32)
    ) -> None:

        super().__init__()

        l0 = get_final_len(res, k1, k2, p1, p2) * c2

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
            torch.nn.Linear(l2, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.stack(x)


def get_final_len(
    res: Tuple[int], k1: int, k2: int, p1: int, p2: int
) -> int:
    """Compute final output size of two sets of (conv3d, maxpool3d) layers
    using conv kernel_size and pool kernel_size of each layer.
    """

    res2 = (math.floor(((r - k1 + 1) - p1) / p1 + 1) for r in res)
    res3 = (math.floor(((r - k2 + 1) - p2) / p2 + 1) for r in res2)
    final_len = 1
    for r in res3:
        final_len *= r
    return final_len


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Tuple[int],
    generator: Callable[[int, int, torch.device, Tuple[int]],
                        Tuple[torch.Tensor]],
    optimizer: torch.optim.Optimizer
) -> None:

    model.train()

    num_batches = num_samples // batch_size
    avg_loss, avg_error = 0, 0

    for X, y in generator(num_batches, batch_size, device, resolution):
        pred = model(X)
        loss = loss_fn(pred, y)

        avg_loss += loss.item()
        avg_error += torch.mean(torch.abs(y - pred) / y, 0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss /= num_batches
    avg_error /= num_batches
    # print(f'avg_error = {avg_error[0]:.5f} {avg_error[1]:.5f}'
    #       f' {avg_error[2]:.5f}')
    # print(f'{loss = :.5f}')


def test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Tuple[int],
    generator: Callable[[int, int, torch.device, Tuple[int]],
                        Tuple[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.nn.Module]:

    model.eval()

    num_batches = num_samples // batch_size
    avg_loss, avg_error = 0, 0
    # Fancy way of using no_grad with testing only, if optimizer is not def
    with torch.no_grad():
        for X, y in generator(num_batches, batch_size, device, resolution):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

    avg_loss /= num_batches
    avg_error /= num_batches
    # print(f'avg_error = {avg_error[0]:.5f} {avg_error[1]:.5f}'
    #       f' {avg_error[2]:.5f}')
    # print(f'{loss = :.5f}')

    return avg_error, avg_loss


def train_test_model(
    conf: Dict[str, Any], checkpoint_dir: Path = None
) -> None:

    device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu'

    model = ConvNeuralNet3D(
        c1=conf['c1'], c2=conf['c2'], k1=conf['k1'], k2=conf['k2'],
        p1=conf['p1'], p2=conf['p2'], l1=conf['l1'], l2=conf['l2'],
        res=conf['resolution']
    ).to(device)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = LogCoshLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

    if checkpoint_dir:
        model_state, optim_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    for epoch in range(20):

        train(
            model=model, loss_fn=loss_fn, device=device,
            num_samples=conf['train_size'], batch_size=conf['batch_size'],
            resolution=conf['resolution'],
            generator=scaling.voxel_image_generator,
            optimizer=optimizer
        )

        (bg_err, bth_err, pe_err), loss = test(
            model=model, loss_fn=loss_fn, device=device,
            num_samples=conf['test_size'], batch_size=conf['batch_size'],
            resolution=conf['resolution'],
            generator=scaling.voxel_image_generator
        )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = Path(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss, bg_err=bg_err, bth_err=bth_err, pe_err=pe_err)


def main():

    # config = {
    #     'batch_size': tune.choice([10, 20, 50, 100]),
    #     'train_size': 70000,
    #     'test_size': 30000,
    #     'lr': 1e-3,
    #     'resolution': (128, 32, 128),
    #     'c1': tune.choice([6, 8, 10, 12]),
    #     'c2': tune.choice([16, 20, 24, 32]),
    #     'k1': tune.choice([1, 3, 5, 7]),
    #     'k2': tune.choice([1, 3, 5, 7]),
    #     'p1': 2,
    #     'p2': 2,
    #     'l1': tune.sample_from(lambda _: 2**random.randint(6, 10)),
    #     'l2': tune.sample_from(lambda _: 2**random.randint(6, 10)),
    # }

    config = {
        'batch_size': tune.choice([10, 20, 50, 100]),
        'train_size': 70000,
        'test_size': 30000,
        'lr': tune.loguniform(1e-4, 1e-1),
        'resolution': (128, 32, 128),
        'c1': 6,
        'c2': 32,
        'k1': 7,
        'k2': 5,
        'p1': 2,
        'p2': 2,
        'l1': 256,
        'l2': 128,
    }

    scheduler = ASHAScheduler(metric='pe_err', mode='min')
    reporter = CLIReporter(
        parameter_columns=['batch_size', 'resolution', 'lr'],
        metric_columns=['loss', 'bg_err', 'bth_err', 'pe_err'],
        max_report_frequency=60,
        max_progress_rows=50,
        print_intermediate_tables=True
    )
    # reporter = CLIReporter(
    #     parameter_columns=[
    #         'batch_size', 'resolution', 'lr', 'c1', 'c2', 'k1', 'k2',
    #         'l1', 'l2'
    #     ],
    #     metric_columns=['loss', 'bg_err', 'bth_err', 'pe_err'],
    #     max_report_frequency=60,
    #     max_progress_rows=50,
    #     print_intermediate_tables=True
    # )

    result = tune.run(
        train_test_model,
        resources_per_trial={'gpu': 1},
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=RAY_PATH,
    )

    best_trial = result.get_best_trial('pe_err', 'min', 'last')
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final loss: {best_trial.last_result["loss"]}')


if __name__ == '__main__':
    main()
