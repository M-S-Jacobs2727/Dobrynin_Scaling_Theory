import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.nn.functional import softplus

import scaling_torch_lib as scaling
from scaling_torch_lib import Resolution

PROJ_PATH = Path("/proj/avdlab/projects/Solutions_ML/")
LOG_PATH = Path(PROJ_PATH, "mike_outputs/")
RAY_PATH = Path(LOG_PATH, "ray_results/")


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + softplus(-2.0 * x) - torch.log(torch.tensor(2.0))

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def custom_MSELoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    Bg, Bth, Pe = y_true.T
    pred_Bg, pred_Bth, pred_Pe = y_pred.T

    mask = Bg < Bth**0.824
    athermal_error = (Bg[mask] - pred_Bg[mask]) ** 2
    athermal_error += (Pe[mask] - pred_Pe[mask]) ** 2
    athermal_error /= 2

    good_error = (Bg[~mask] - pred_Bg[~mask]) ** 2
    good_error += (Bth[~mask] - pred_Bth[~mask]) ** 2
    good_error += (Pe[~mask] - pred_Pe[~mask]) ** 2
    good_error /= 3

    return torch.mean(torch.cat((athermal_error, good_error)))


class CustomMSELoss(torch.nn.Module):
    """A custom implementation of the mean squared error class that accounts
    for the existence of athermal solutions, for which the Bth parameter is
    unused. When computing the loss, for any athermal systems
    (i.e., Bg < Bth**0.824), we only compute the loss for the Bg and Pe params.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return custom_MSELoss(y_pred, y_true)


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


def get_final_len(res: Resolution, k: int, p: int) -> Resolution:
    return Resolution(
        math.floor(((res.phi - k + 1) - p) / p + 1),
        math.floor(((res.Nw - k + 1) - p) / p + 1),
        math.floor(((res.eta_sp - k + 1) - p) / p + 1) if res.eta_sp else 0,
    )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Resolution,
    generator: Callable,
    optimizer: torch.optim.Optimizer,
) -> None:

    model.train()

    num_batches = num_samples // batch_size
    # avg_loss, avg_error = 0, 0

    for X, y in generator(num_batches, batch_size, device, resolution):
        pred = model(X)
        loss = loss_fn(pred, y)

        # avg_loss += loss.item()
        # avg_error += torch.mean(torch.abs(y - pred) / y, 0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # avg_loss /= num_batches
    # avg_error /= num_batches
    # print(f'avg_error = {avg_error[0]:.5f} {avg_error[1]:.5f}'
    #       f' {avg_error[2]:.5f}')
    # print(f'{loss = :.5f}')


def test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Resolution,
    generator: Callable,
) -> Tuple[torch.Tensor, float]:

    model.eval()

    num_batches = num_samples // batch_size
    avg_loss, avg_error = 0, torch.zeros(3)
    with torch.no_grad():
        for X, y in generator(num_batches, batch_size, device, resolution):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += float(loss.item())
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

    avg_loss /= num_batches
    avg_error /= num_batches
    # print(f'avg_error = {avg_error[0]:.5f} {avg_error[1]:.5f}'
    #       f' {avg_error[2]:.5f}')
    # print(f'{loss = :.5f}')

    return avg_error, avg_loss


def train_test_model(conf: Dict[str, Any], checkpoint_dir: Path) -> None:

    if conf["resolution"].eta_sp:
        generator = scaling.voxel_image_generator
    else:
        generator = scaling.surface_generator

    device = torch.device("cuda:0")  # if torch.cuda.is_available() else 'cpu'

    model = LinearNeuralNet(
        res=conf["resolution"], l1=conf["l1"], l2=conf["l2"], l3=conf["l3"]
    ).to(device)

    loss_fn = CustomMSELoss()
    # loss_fn = LogCoshLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"])

    if checkpoint_dir:
        model_state, optim_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    for epoch in range(conf["epochs"]):

        train(
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_samples=conf["train_size"],
            batch_size=conf["batch_size"],
            resolution=conf["resolution"],
            generator=generator,
            optimizer=optimizer,
        )

        (bg_err, bth_err, pe_err), loss = test(
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_samples=conf["test_size"],
            batch_size=conf["batch_size"],
            resolution=conf["resolution"],
            generator=generator,
        )
        bg_err = bg_err.cpu()
        bth_err = bth_err.cpu()
        pe_err = pe_err.cpu()

        if (epoch + 1) % 5 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss, bg_err=bg_err, bth_err=bth_err, pe_err=pe_err)


def main():

    config = {
        "epochs": 200,
        "batch_size": 50,
        "train_size": 40_000,
        "test_size": 5000,
        "lr": 3e-4,
        "resolution": Resolution(512, 512),
        "l1": tune.choice([1024, 2048]),
        "l2": tune.choice([256, 512, 1024]),
        "l3": tune.choice([256, 512, 1024]),
    }

    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=100)
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "l3"],
        metric_columns=["bg_err", "bth_err", "pe_err", "loss"],
        max_report_frequency=60,
        max_progress_rows=50,
    )

    result = tune.run(
        train_test_model,
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=50,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=str(RAY_PATH),
    )

    best_trial = result.get_best_trial("loss", "min")
    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial config: {best_trial.config}")
    print(f'Best trial final loss: {best_trial.last_result["loss"]}')

    df: pd.DataFrame = result.dataframe()
    print(df.sort_values(by="loss"))


if __name__ == "__main__":
    main()
