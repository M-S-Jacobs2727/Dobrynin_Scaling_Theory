"""This script takes a completed model and evaluates it using more theoretical data.
In addition to the errors of the predicted Bg and Bth values, the Pe values are also
predicted using the physical knowledge and a basic cubic fit (see the `fit_pe` func).

TODO: turn this into a notebook? (.ipynb)
"""
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as generators
import theoretical_nn_training.models as models
from theoretical_nn_training.configuration import NNConfig


def unnormalize_params(
    Bg: np.ndarray,
    Bth: np.ndarray,
    Pe: np.ndarray,
    bg_range: data.Range,
    bth_range: data.Range,
    pe_range: data.Range,
) -> Tuple[np.ndarray, ...]:
    """Inverts simple linear normalization."""
    Bg = Bg * (bg_range.max - bg_range.min) + bg_range.min
    Bth = Bth * (bth_range.max - bth_range.min) + bth_range.min
    Pe = Pe * (pe_range.max - pe_range.min) + pe_range.min
    return Bg, Bth, Pe


def test_accuracy(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: NNConfig,
    generator: generators.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:

    num_batches = config.test_size // config.batch_size

    all_true = np.zeros((num_batches, config.batch_size, config.layer_sizes[-1]))
    all_pred = np.zeros((num_batches, config.batch_size, config.layer_sizes[-1]))
    all_surfaces = np.zeros(
        (num_batches, config.batch_size, config.resolution.phi, config.resolution.Nw)
    )

    model.eval()

    cum_losses = torch.zeros((3,))
    with torch.no_grad():
        for b, (surfaces, features) in enumerate(generator(num_batches)):
            pred = model(surfaces)
            losses = loss_fn(pred, features)

            all_true[b] = features.cpu()
            all_pred[b] = pred.cpu()
            all_surfaces[b] = surfaces.cpu()

            cum_losses += losses.mean(dim=0).cpu()

    return (
        all_true.reshape(num_batches * config.batch_size, config.layer_sizes[-1]),
        all_pred.reshape(num_batches * config.batch_size, config.layer_sizes[-1]),
        all_surfaces.reshape(
            num_batches * config.batch_size,
            config.resolution.phi * config.resolution.Nw,
        ),
        cum_losses / num_batches,
    )


def fit_func(x: np.ndarray, Pe: float) -> np.ndarray:
    return x * (1 + x**2 / Pe**4)


def fit_func_jacobian(x: np.ndarray, Pe: float) -> np.ndarray:
    return -4 * x**3 / Pe**5


def fit_pe(
    bg_pred: np.ndarray,
    bth_pred: np.ndarray,
    phi_mesh: np.ndarray,
    Nw_mesh: np.ndarray,
    all_surfaces: np.ndarray,
) -> np.ndarray:
    pe_fit = np.zeros_like(bg_pred)
    for i, (bg, bth, eta_sp, phi, Nw) in enumerate(
        zip(bg_pred, bth_pred, phi_mesh, Nw_mesh, all_surfaces)
    ):
        print(i)
        lamda_g = np.maximum(
            (bth / bg) ** (2 / 3 / (2 * 0.588 - 1)) / bth**4,
            np.minimum(1, phi ** (2 / 3) / bth**4),
        )
        lamda = np.maximum(1, phi / bth**4)
        g = np.minimum((bg**3 / phi) ** (1 / (3 * 0.588 - 1)), bth**6 / phi**2)
        y = lamda * eta_sp
        x = Nw / g / lamda_g

        popt, pcov = curve_fit(
            fit_func, x, y, (8,), bounds=(0, 30), jac=fit_func_jacobian
        )
        pe_fit[i] = popt[0]
        print(f"{popt[0]} +/- {pcov[0]}")

    return pe_fit


def plot(
    pred: np.ndarray, true: np.ndarray, title: str, color: str, athermal: np.ndarray
):

    plt.figure(title)
    plot_min = min(pred.min(), true.min())
    plot_max = max(pred.max(), true.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], "k-", lw=3)
    plt.plot(
        true[athermal],
        pred[athermal],
        marker="o",
        mec="xkcd:grey",
        mfc="None",
        mew=1,
        ls="None",
    )
    plt.plot(
        true[~athermal],
        pred[~athermal],
        marker="o",
        mec=color,
        mfc="None",
        mew=1,
        ls="None",
    )
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])


def main() -> None:
    # Set device, config, model, loss function, and generator

    config = NNConfig(
        "theoretical_nn_training/configurations/sample_config_rcs_512.yaml"
    )
    if (
        config.channels is None
        or config.kernel_sizes is None
        or config.pool_sizes is None
    ):
        raise RuntimeError("Invalid configuration.")

    checkpoint = torch.load("../mike_outputs/complex_out/model_and_optimizer")
    model = models.ConvNeuralNet2D(
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        pool_sizes=config.pool_sizes,
        layer_sizes=config.layer_sizes,
        resolution=config.resolution,
    ).to(config.device)
    model.load_state_dict(checkpoint["model"])

    loss_fn = torch.nn.MSELoss(reduction="none")

    generator = generators.SurfaceGenerator(config)

    # Test and collect data
    all_true, all_pred, all_surfaces, losses = test_accuracy(
        model=model, loss_fn=loss_fn, config=config, generator=generator
    )

    print(f"{losses = }")

    bg_true, bth_true, pe_true = unnormalize_params(
        all_true[:, 0],
        all_true[:, 1],
        all_true[:, 2],
        config.bg_range,
        config.bth_range,
        config.pe_range,
    )

    if all_pred.shape[2] == 3:
        bg_pred, bth_pred, pe_pred = unnormalize_params(
            all_pred[:, 0],
            all_pred[:, 1],
            all_pred[:, 2],
            config.bg_range,
            config.bth_range,
            config.pe_range,
        )
    elif all_pred.shape[2] == 2:
        bg_pred, bth_pred, _ = unnormalize_params(
            all_pred[:, 0],
            all_pred[:, 1],
            np.zeros(1),
            config.bg_range,
            config.bth_range,
            config.pe_range,
        )
        pe_pred = fit_pe(
            bg_pred=bg_pred,
            bth_pred=bth_pred,
            phi_mesh=generator.phi_mesh.numpy(),
            Nw_mesh=generator.Nw_mesh.numpy(),
            all_surfaces=all_surfaces,
        )
    else:
        raise

    athermal = bg_true < bth_true**0.824

    # Show parity plots
    plot(bg_pred, bg_true, "Bg", "r", athermal)
    plot(bth_pred, bth_true, "Bth", "g", athermal)
    plot(pe_pred, pe_true, "Pe", "b", athermal)
    plt.show()

    # Save true and predicted values
    np.savetxt(
        "../mike_outputs/true_pred_vals.csv",
        np.stack((bg_true, bg_pred, bth_true, bth_pred, pe_true, pe_pred), axis=1),
        header="Bg_true,Bg_pred,Bth_true,Bth_pred,Pe_true,Pe_pred",
        delimiter=",",
        comments="",
        fmt="%.3f",
    )


if __name__ == "__main__":
    main()
