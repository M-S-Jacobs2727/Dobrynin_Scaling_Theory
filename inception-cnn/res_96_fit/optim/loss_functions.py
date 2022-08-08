import torch
from torch.nn.functional import softplus
import scaling_torch_lib as mike

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + softplus(-2.0 * x) - torch.log(torch.tensor(2.0))

    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

def custom_MSE_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    Bg, Bth, _ = mike.unnormalize_params(y_true)
    mask = Bg > Bth**0.824

    #Bg2, Bth2, _ = mike.unnormalize_params(y_pred)
    #mask2 = Bg2 > Bth2**0.824

    Bg_loss = torch.mean((y_true[mask][:, 0] - y_pred[mask][:, 0]) ** 2)
    Bth_loss = torch.mean((y_true[mask][:, 1] - y_pred[mask][:, 1]) ** 2)
    Pe_loss = torch.mean((y_true[mask][:, 2] - y_pred[mask][:, 2]) ** 2)

    loss = torch.sum(Bg_loss + Bth_loss + Pe_loss)

    return loss

class CustomMSELoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor
            ) -> torch.Tensor:
        return custom_MSE_loss(y_pred, y_true)

def custom_MAE_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    Bg, Bth, _ = mike.unnormalize_params(y_true)
    mask = Bg > Bth**0.824

    #Bg2, Bth2, _ = mike.unnormalize_params(y_pred)
    #mask2 = Bg2 < Bth2**0.824


    Bth_loss = torch.mean(torch.abs(y_true[mask][:, 1] - y_pred[mask][:, 1]))

    loss = torch.sum(torch.mean(torch.abs(y_true[:, 0] - y_pred[:, 0])) + Bth_loss + torch.mean(torch.abs(y_true[:,2] - y_pred[:,2])))

    return loss

class CustomMAELoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor
            ) -> torch.Tensor:
        return custom_MSE_loss(y_pred, y_true)

def custom_MSE_loss_old(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    loss = (y_true - y_pred) ** 2

    # Test the athermal condition. The solvent is too good for thermal fluctuations.
    Bg, Bth, _ = mike.unnormalize_params2(*(y_true.T))
    mask = Bg < Bth**0.824

    athermal_loss = (loss[mask][:, 0] + loss[mask][:, 2]) / 2
    good_loss = torch.mean(loss[~mask], dim=1)

    return torch.mean(torch.cat((athermal_loss, good_loss)))

class MSECustomLoss_old(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor
            ) -> torch.Tensor:
        return custom_MSE_loss_old(y_pred, y_true)
