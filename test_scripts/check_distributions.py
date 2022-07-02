import matplotlib.pyplot as plt
import theoretical_nn_training.configuration as configuration
import theoretical_nn_training.data_processing as data
import torch


def sample_and_plot(
    feature_range: data.Range, batch_size: int, feature_name: str, device: torch.device
) -> None:
    dist = data.feature_distribution(feature_range, device)
    values = dist.sample(torch.Size((batch_size,)))

    plt.figure(feature_name)
    plt.hist(values.numpy(), bins=20)


def main() -> None:
    config = configuration.read_config_from_file(
        "theoretical_nn_training/configurations/mixed_config_512.yaml"
    )
    config.batch_size = 10000

    sample_and_plot(config.bg_range, config.batch_size, "Bg", config.device)
    sample_and_plot(config.bth_range, config.batch_size, "Bth", config.device)
    sample_and_plot(config.pe_range, config.batch_size, "Pe", config.device)
    plt.show()


if __name__ == "__main__":
    main()
