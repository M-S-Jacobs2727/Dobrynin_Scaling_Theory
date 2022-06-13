from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as generators
import theoretical_nn_training.loss_funcs as loss_funcs
import theoretical_nn_training.models as models


def test_accuracy(
    device: torch.device,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    num_samples: int,
    config: data.NNConfig,
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    num_batches = num_samples // config.batch_size

    all_y = torch.zeros((num_batches, config.batch_size, 3))
    all_pred = torch.zeros((num_batches, config.batch_size, 3))

    model.eval()

    cum_loss = 0
    with torch.no_grad():
        for b, (X, y) in enumerate(
            generators.voxel_image_generator(num_batches, device, config)
        ):
            pred = model(X)
            loss = loss_fn(pred, y)

            all_y[b] = y
            all_pred[b] = pred

            cum_loss += float(loss.item())

    return all_y, all_pred, cum_loss / num_samples


def main() -> None:
    device = torch.device("cuda:0")

    config = data.NNConfig("theoretical_nn_training/configurations/sample_config.yaml")

    model_state, _ = torch.load("../mike_outputs/sample_checkpoint")
    if not (config.channels and config.kernel_sizes and config.pool_sizes):
        raise ValueError(
            "Need full convolutional neural network configuration."
            "Missing one of channels, kernel_sizes, or pool_sizes."
        )
    if not config.resolution.eta_sp:
        raise ValueError("Need 3D resolution. Missing eta_sp dimension.")
    model = models.ConvNeuralNet3D(
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        pool_sizes=config.pool_sizes,
        layer_sizes=config.layer_sizes,
        resolution=config.resolution,
    ).to(device)
    model.load_state_dict(model_state)

    loss_fn = loss_funcs.LogCoshLoss()

    y, pred, loss = test_accuracy(device, model, loss_fn, config.test_size, config)

    print(loss)
    print(torch.abs(y - pred) / y)

    bg_true = y[:, :, 0]
    bth_true = y[:, :, 1]
    pe_true = y[:, :, 2]
    bg_pred = pred[:, :, 0]
    bth_pred = pred[:, :, 1]
    pe_pred = pred[:, :, 2]

    plt.figure("Bg")
    plt.plot([0, 1], [0, 1], "k-", lw=3)
    plt.plot(
        bg_true.flatten(),
        bg_pred.flatten(),
        marker="o",
        mec="r",
        mfc="None",
        mew=1,
        ls="None",
    )

    plt.figure("Bth")
    plt.plot([0, 1], [0, 1], "k-", lw=3)
    plt.plot(
        bth_true.flatten(),
        bth_pred.flatten(),
        marker="o",
        mec="b",
        mfc="None",
        mew=1,
        ls="None",
    )

    plt.figure("Pe")
    plt.plot([0, 1], [0, 1], "k-", lw=3)
    plt.plot(
        pe_true.flatten(),
        pe_pred.flatten(),
        marker="o",
        mec="g",
        mfc="None",
        mew=1,
        ls="None",
    )
    plt.show()

    bg_true, bth_true, pe_true = (
        bg_true.flatten(),
        bth_true.flatten(),
        pe_true.flatten(),
    )
    bg_pred, bth_pred, pe_pred = (
        bg_pred.flatten(),
        bth_pred.flatten(),
        pe_pred.flatten(),
    )
    bg_true, bth_true, pe_true = data.unnormalize_params(
        bg_true, bth_true, pe_true, config.bg_range, config.bth_range, config.pe_range
    )
    bg_pred, bth_pred, pe_pred = data.unnormalize_params(
        bg_pred, bth_pred, pe_pred, config.bg_range, config.bth_range, config.pe_range
    )

    np.savetxt(
        "../mike_outputs/true_vals",
        torch.stack(
            (bg_true, bg_pred, bth_true, bth_pred, pe_true, pe_pred), dim=1
        ).numpy(),
        header="Bg,Bg_pred,Bth,Bth_pred,Pe,Pe_pred",
        delimiter=",",
        comments="",
        fmt="%.3f",
    )


if __name__ == "__main__":
    main()
