from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import gen_and_train as nn
import scaling_torch_lib as scaling


def test_accuracy(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    res: Tuple[int],
) -> Tuple[torch.Tensor]:

    num_batches = num_samples // batch_size

    all_y = torch.zeros((num_batches, batch_size, 3))
    all_pred = torch.zeros((num_batches, batch_size, 3))

    model.eval()

    cum_loss = 0
    with torch.no_grad():
        for b, (X, y) in enumerate(
            scaling.voxel_image_generator(num_batches, batch_size, device, res)
        ):
            pred = model(X)
            loss = loss_fn(pred, y)

            all_y[b] = y
            all_pred[b] = pred

            cum_loss += loss.item()

    return all_y, all_pred, cum_loss / num_samples


def main() -> None:
    device = torch.device("cuda:0")
    model_state, _ = torch.load("../mike_outputs/sample_checkpoint")
    model = nn.ConvNeuralNet3D(
        c1=6, k1=7, p1=2, c2=32, k2=5, p2=2, l1=256, l2=128, res=(128, 32, 128)
    ).to(device)
    model.load_state_dict(model_state)

    y, pred, loss = test_accuracy(
        model, nn.log_cosh_loss, device, 2000, 100, (128, 32, 128)
    )

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
    bg_true, bth_true, pe_true = scaling.unnormalize_params(
        torch.stack((bg_true, bth_true, pe_true), dim=1)
    )
    bg_pred, bth_pred, pe_pred = scaling.unnormalize_params(
        torch.stack((bg_pred, bth_pred, pe_pred), dim=1)
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
