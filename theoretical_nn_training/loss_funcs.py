"""Two custom loss functions, `LogCoshLoss` is taken from a stackexchange post (cite)
and `CustomMSELoss` applies an MSE loss without punishing in the case of an athermal
solvent ($B_{g} < B_{th}^{0.824}$).
"""
from typing import Optional

import torch
from torch.nn.functional import softplus

import theoretical_nn_training.data_processing as data


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + softplus(-2.0 * x) - torch.log(torch.tensor(2.0))

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def _custom_MSE_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    bg_range: data.Range,
    bth_range: data.Range,
    pe_range: data.Range,
) -> torch.Tensor:

    loss = (y_true - y_pred) ** 2

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = data.unnormalize_features(
        y_true[:, 0], y_true[:, 1], y_true[:, 2], bg_range, bth_range, pe_range
    )
    athermal = Bg < Bth**0.824

    return torch.tensor(
        [
            torch.mean(loss[:, 0]),
            torch.mean(loss[~athermal][:, 1]),
            torch.mean(loss[:, 2]),
        ],
        requires_grad=True,
        device=y_true.device,
    )


def _custom_MSE_loss_no_Pe(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    bg_range: data.Range,
    bth_range: data.Range,
) -> torch.Tensor:

    loss = (y_true[:, :2] - y_pred) ** 2

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = data.unnormalize_features(
        y_true[:, 0],
        y_true[:, 1],
        torch.tensor([0]),
        bg_range,
        bth_range,
        data.Range(0, 1),
    )
    athermal = Bg < Bth**0.824

    return torch.tensor(
        [torch.mean(loss[:, 0]), torch.mean(loss[~athermal][:, 1])],
        requires_grad=True,
        device=y_true.device,
    )


class CustomMSELoss(torch.nn.Module):
    """A custom implementation of the mean squared error class that accounts
    for the existence of athermal solutions (i.e., $B_g < B_{th}^{0.824}$), for
    which the $B_{th}$ parameter is impossible to detect. When computing the loss,
    for any athermal systems, we only compute the loss for the $B_g$ parameter and,
    if applicable, the $P_e$ parameter.

    Input:
        `bg_range` (`data_processing.Range`) : Used to compute the true values of
            the $B_g$ parameter from the normalized values.
        `bth_range` (`data_processing.Range`) : Used to compute the true values of
            the $B_{th}$ parameter from the normalized values.
        `pe_range` (`data_processing.Range`, optional) : Used to compute the true
            values of the packing number $P_e$ from the normalized values.
        `mode` (`str`, default 'mean') : Either 'mean' or 'none'. If 'mean', the
            loss values of the parameters are averaged, and a singlton Tensor is
            returned. If 'none', the loss values of each parameter are returned in a
            length 3 Tensor if `pe_range` is given or a length 2 Tensor otherwise.
    """

    def __init__(
        self,
        bg_range: data.Range,
        bth_range: data.Range,
        pe_range: Optional[data.Range] = None,
        mode: str = "mean",
    ) -> None:
        if mode not in ["none", "mean"]:
            raise SyntaxError(
                f"Expected a mode of either `none` or `mean`, not `{mode}`"
            )
        super().__init__()
        self.bg_range = bg_range
        self.bth_range = bth_range
        self.pe_range = pe_range
        self._mode = mode

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self._mode == "none" and self.pe_range:
            return _custom_MSE_loss(
                y_pred, y_true, self.bg_range, self.bth_range, self.pe_range
            )
        elif self._mode == "mean" and self.pe_range:
            return torch.mean(
                _custom_MSE_loss(
                    y_pred, y_true, self.bg_range, self.bth_range, self.pe_range
                )
            )
        elif self._mode == "none":
            return _custom_MSE_loss_no_Pe(y_pred, y_true, self.bg_range, self.bth_range)
        elif self._mode == "mean":
            return torch.mean(
                _custom_MSE_loss_no_Pe(y_pred, y_true, self.bg_range, self.bth_range)
            )
        else:
            raise SyntaxError(
                f"Expected a mode of either `none` or `mean`, not `{self._mode}`"
            )
