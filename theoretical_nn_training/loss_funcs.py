import torch
from torch.nn.functional import softplus

from theoretical_nn_training.data_processing import Param, unnormalize_params


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + softplus(-2.0 * x) - torch.log(torch.tensor(2.0))

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def custom_MSE_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    bg_param: Param,
    bth_param: Param,
    pe_param: Param,
) -> torch.Tensor:
    loss = (y_true - y_pred) ** 2

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = unnormalize_params(
        y_true[:, 0], y_true[:, 1], y_true[:, 2], bg_param, bth_param, pe_param
    )
    mask = Bg < Bth**0.824

    athermal_loss = (loss[mask][:, 0] + loss[mask][:, 2]) / 2
    good_loss = torch.mean(loss[~mask], dim=1)

    return torch.mean(torch.cat((athermal_loss, good_loss)))


class CustomMSELoss(torch.nn.Module):
    """A custom implementation of the mean squared error class that accounts
    for the existence of athermal solutions, for which the Bth parameter is
    unused. When computing the loss, for any athermal systems
    (i.e., Bg < Bth**0.824), we only compute the loss for the Bg and Pe params.
    """

    def __init__(self, bg_param: Param, bth_param: Param, pe_param: Param) -> None:
        super().__init__()
        self.bg_param = bg_param
        self.bth_param = bth_param
        self.pe_param = pe_param

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return custom_MSE_loss(
            y_pred, y_true, self.bg_param, self.bth_param, self.pe_param
        )
