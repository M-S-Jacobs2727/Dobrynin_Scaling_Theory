"""This script takes a completed model and evaluates it using more theoretical data.
In addition to the errors of the predicted Bg and Bth values, the Pe values are also
predicted using the physical knowledge and a basic cubic fit (see the `fit_pe` func).

TODO: turn this into a notebook? (.ipynb)
"""
import argparse
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import theoretical_nn_training.configuration as configuration
import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as generators
import theoretical_nn_training.models as models
import torch
from scipy.optimize import curve_fit


def unnormalize_feature(
    feature: np.ndarray,
    feature_range: data.Range,
) -> np.ndarray:
    """Inverts simple linear normalization."""
    return feature * (feature_range.max - feature_range.min) + feature_range.min


def test_accuracy(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: configuration.NNConfig,
    generator: generators.Generator,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:

    num_batches = num_samples // config.batch_size

    all_true = np.zeros((num_batches, config.batch_size, config.layer_sizes[-1]))
    all_pred = np.zeros((num_batches, config.batch_size, config.layer_sizes[-1]))
    all_surfaces = np.zeros(
        (num_batches, config.batch_size, config.resolution.phi, config.resolution.Nw)
    )

    model.eval()

    cum_losses = torch.zeros((3,) if config.mode is data.Mode.MIXED else (2,))
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


def fit_func(Nw_over_g: np.ndarray, Pe: float) -> np.ndarray:
    return Nw_over_g * (1 + Nw_over_g**2 / Pe**4)


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


def plot(pred: np.ndarray, true: np.ndarray, title: str, color: str):

    error = np.mean(np.abs(pred - true) / true)
    std_dev = np.std(np.abs(pred - true) / true)
    plt.figure(title)
    plot_min = min(pred.min(), true.min())
    plot_max = max(pred.max(), true.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], "k-", lw=3)
    plt.plot(
        true,
        pred,
        marker="o",
        mec=color,
        mfc="None",
        mew=1,
        ls="None",
        label=f"{error:.2g} +/- {std_dev:.2g}",
    )
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])
    plt.legend()


def main() -> None:

    # Parse arguments for logfile and verbosity (logging.debug vs. logging.info)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str)
    parser.add_argument(
        "-m",
        "--modelfile",
        type=str,
        help="The location of a file containing the states of the model and optimizer"
        " as a dictionary. To be read by `torch.load`.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="Number of samples to test. If unset, read from config.",
    )
    args = parser.parse_args()

    # Get configuration file from command line
    config_filename = Path(args.config_filename)

    if not config_filename.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_filename.absolute()}"
        )

    # Set config, loss function, and generator
    config = configuration.read_config_from_file(args.config_filename)
    if (
        config.channels is None
        or config.kernel_sizes is None
        or config.pool_sizes is None
    ):
        raise RuntimeError("Invalid configuration.")
    if config.mode not in [data.Mode.GOOD, data.Mode.THETA, data.Mode.MIXED]:
        raise ValueError(f"Invalid mode setting in config: {config.mode}")

    loss_fn = torch.nn.MSELoss(reduction="none")
    generator = generators.SurfaceGenerator(config)

    # Read model from file
    modelfile = Path(args.modelfile)
    if not modelfile.is_file():
        raise FileNotFoundError(f"Model file not found: {modelfile.absolute}")

    checkpoint = torch.load(modelfile, map_location=config.device)
    model = models.ConvNeuralNet2D(
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        pool_sizes=config.pool_sizes,
        layer_sizes=config.layer_sizes,
    ).to(config.device)
    model.load_state_dict(checkpoint["model"])

    # Set number of samples from command line if provided, config file if not
    num_samples = args.num_samples if args.num_samples else config.test_size

    # Test and collect data
    normed_true, normed_pred, all_surfaces, losses = test_accuracy(
        model=model,
        loss_fn=loss_fn,
        config=config,
        generator=generator,
        num_samples=num_samples,
    )

    print(f"{losses = }")

    if config.mode is data.Mode.GOOD:
        bg_true = unnormalize_feature(normed_true[:, 0], config.bg_range)
        bg_pred = unnormalize_feature(normed_pred[:, 0], config.bg_range)
        pe_true = unnormalize_feature(normed_true[:, 1], config.pe_range)
        pe_pred = unnormalize_feature(normed_pred[:, 1], config.pe_range)
        plot(bg_pred, bg_true, "Bg", "r")
        plot(pe_pred, pe_true, "Pe", "b")
        out_data = np.stack((bg_true, bg_pred, pe_true, pe_pred), axis=1)
        header = "Bg_true,Bg_pred,Pe_true,Pe_pred"
    elif config.mode is data.Mode.THETA:
        bth_true = unnormalize_feature(normed_true[:, 0], config.bth_range)
        bth_pred = unnormalize_feature(normed_pred[:, 0], config.bth_range)
        pe_true = unnormalize_feature(normed_true[:, 1], config.pe_range)
        pe_pred = unnormalize_feature(normed_pred[:, 1], config.pe_range)
        plot(bth_pred, bth_true, "Bth", "g")
        plot(pe_pred, pe_true, "Pe", "b")
        out_data = np.stack((bth_true, bth_pred, pe_true, pe_pred), axis=1)
        header = "Bth_true,Bth_pred,Pe_true,Pe_pred"
    else:
        bg_true = unnormalize_feature(normed_true[:, 0], config.bg_range)
        bg_pred = unnormalize_feature(normed_pred[:, 0], config.bg_range)
        bth_true = unnormalize_feature(normed_true[:, 1], config.bth_range)
        bth_pred = unnormalize_feature(normed_pred[:, 1], config.bth_range)
        pe_true = unnormalize_feature(normed_true[:, 2], config.pe_range)
        pe_pred = unnormalize_feature(normed_pred[:, 2], config.pe_range)
        plot(bg_pred, bg_true, "Bg", "r")
        plot(bth_pred, bth_true, "Bth", "g")
        plot(pe_pred, pe_true, "Pe", "b")
        out_data = np.stack(
            (bg_true, bg_pred, bth_true, bth_pred, pe_true, pe_pred), axis=1
        )
        header = "Bg_true,Bg_pred,Bth_true,Bth_pred,Pe_true,Pe_pred"

    plt.show()

    # Save true and predicted values
    np.savetxt(
        modelfile.parent / "true_pred_vals.csv",
        out_data,
        header=header,
        delimiter=",",
        comments="",
        fmt="%.3f",
    )


if __name__ == "__main__":
    main()
