import matplotlib.pyplot as plt
import theoretical_nn_training.data_processing as data
from theoretical_nn_training.configuration import NNConfig


def sample_and_plot(
    feature_range: data.Range, batch_size: int, feature_name: str
) -> None:
    dist = data.feature_distribution(feature_range, batch_size)
    values = dist.sample()

    plt.figure(feature_name)
    plt.hist(values.numpy(), bins=20)


def main() -> None:
    config = NNConfig("theoretical_nn_training/configurations/mixed_config_512.yaml")
    config.batch_size = 1000

    sample_and_plot(config.bg_range, config.batch_size, "Bg")
    sample_and_plot(config.bth_range, config.batch_size, "Bth")
    sample_and_plot(config.pe_range, config.batch_size, "Pe")
    plt.show()


if __name__ == "__main__":
    main()
