"""Two custom loss functions, `LogCoshLoss` is taken from a stackexchange post (cite)
and `CustomMSELoss` applies an MSE loss without punishing in the case of an athermal
solvent ($B_{g} < B_{th}^{0.824}$).
"""

import logging

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
    logger: logging.Logger,
) -> torch.Tensor:

    if y_pred.size(1) == 2:
        loss = (y_true[:, :2] - y_pred) ** 2
    elif y_pred.size(1) == 3:
        loss = (y_true - y_pred) ** 2
    else:
        logger.exception(
            f"Invalid number of features in loss function: {y_pred.size(1)}."
            " Should be 2 or 3."
        )
        raise

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = data.unnormalize_features(
        y_true[:, 0], y_true[:, 1], y_true[:, 2], bg_range, bth_range, pe_range
    )
    athermal = Bg < Bth**0.824

    loss[:, 1][athermal] = 0

    avg_loss = torch.mean(loss, dim=0)
    avg_loss[1] *= y_pred.size(0) / torch.sum(~athermal)

    return avg_loss


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
        `pe_range` (`data_processing.Range`) : Used to compute the true values of
            the packing number $P_e$ from the normalized values.
        `mode` (`str`, default 'mean') : One of 'mean', 'sum', or 'none'. If 'mean', the
            loss values of the parameters are averaged, and a singlton Tensor is
            returned. If 'sum', they are added together instead. If 'none', the loss
            values of each parameter are returned in a 1D tensor the same length as
            y_pred.
    """

    def __init__(
        self,
        bg_range: data.Range,
        bth_range: data.Range,
        pe_range: data.Range,
        mode: str = "mean",
    ) -> None:
        logger = logging.getLogger("__main__")
        if mode not in ["none", "mean", "sum"]:
            logger.exception(
                f"Expected a mode of 'none', 'mean', or 'sum' not '{mode}'"
            )
            raise

        super().__init__()

        if mode == "none":
            self.func = lambda y_pred, y_true: _custom_MSE_loss(
                y_pred,
                y_true,
                bg_range,
                bth_range,
                pe_range,
                logger,
            )
        elif mode == "sum":
            self.func = lambda y_pred, y_true: torch.sum(
                _custom_MSE_loss(
                    y_pred,
                    y_true,
                    bg_range,
                    bth_range,
                    pe_range,
                    logger,
                )
            )
        elif mode == "mean":
            self.func = lambda y_pred, y_true: torch.mean(
                _custom_MSE_loss(
                    y_pred,
                    y_true,
                    bg_range,
                    bth_range,
                    pe_range,
                    logger,
                )
            )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.func(y_pred, y_true)
