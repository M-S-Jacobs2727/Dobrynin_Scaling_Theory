from typing import Tuple, Union

import torch
from torch.nn.functional import softplus

import theoretical_nn_training.data_processing as data

# TODO: Add loss funcs for non-Pe training


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
    bg_range: data.Range,
    bth_range: data.Range,
    pe_range: data.Range,
) -> torch.Tensor:
    loss = (y_true - y_pred) ** 2

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = data.unnormalize_params(
        y_true[:, 0], y_true[:, 1], y_true[:, 2], bg_range, bth_range, pe_range
    )
    mask = Bg < Bth**0.824

    Bg_loss = torch.mean(loss[:, 0])
    Bth_loss = torch.mean(loss[mask][:, 1])
    Pe_loss = torch.mean(loss[:, 2])

    return torch.tensor([Bg_loss, Bth_loss, Pe_loss])


class CustomMSELoss(torch.nn.Module):
    def __init__(
        self,
        bg_range: data.Range,
        bth_range: data.Range,
        pe_range: data.Range,
        mode: str = "mean",
    ) -> None:
        """A custom implementation of the mean squared error class that accounts
        for the existence of athermal solutions (i.e., Bg < Bth**0.824), for which the
        Bth parameter is impossible to detect. When computing the loss, for any athermal
        systems, we only compute the loss for the Bg and Pe params.
        """
        if mode not in ["none", "mean"]:
            raise SyntaxError(f"Expected a mode of either `none` or `mean`, not {mode}")
        super().__init__()
        self.bg_range = bg_range
        self.bth_range = bth_range
        self.pe_range = pe_range
        self._mode = mode

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self._mode == "none":
            return custom_MSE_loss(
                y_pred, y_true, self.bg_range, self.bth_range, self.pe_range
            )
        elif self._mode == "mean":
            return torch.mean(
                custom_MSE_loss(
                    y_pred, y_true, self.bg_range, self.bth_range, self.pe_range
                )
            )
        else:
            raise SyntaxError(
                f"Expected a mode of either `none` or `mean`, not {self._mode}"
            )
