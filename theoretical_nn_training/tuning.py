from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.loss_funcs as loss_funcs
import theoretical_nn_training.models as models
import theoretical_nn_training.training as training


@dataclass
class AllFailedError(Exception):
    result: tune.ExperimentAnalysis


def run(
    config: Dict[str, Any],
    checkpoint_dir: str,
    my_config: data.NNConfig,
    device: torch.device,
) -> None:
    # config is not currently used explicitly; see note below
    """Run a single configuration of settings for the neural network.

    Input:
        `config` (dictionary of `str` to Any): Contains the configuration parameters
            that allows ray.tune to track and report the progress and quality of the
            run. (see tune.report(...) below).
        `checkpoint_dir` (`str`): The directory used by ray.tune to periodically write
            checkpoints of the neural network training for each configuration.
        `my_config` (`data.NNConfig`): This is the actual configuration to use
            throughout the codebase. `NNConfig` is a dataclass (see
            https://docs.python.org/3/library/dataclasses.html), which makes processing
            and reading configurations easier than dictionaries.
    """

    model = models.LinearNeuralNet(
        resolution=my_config.resolution, layer_sizes=my_config.layer_sizes
    ).to(device)

    loss_fn = loss_funcs.CustomMSELoss(
        my_config.bg_range, my_config.bth_range, my_config.pe_range
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config.learning_rate)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(my_config.epochs):

        training.train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=my_config,
            device=device,
        )

        error_ratios, loss = training.test(
            model=model, loss_fn=loss_fn, config=my_config, device=device
        )
        # The error ratios are defined as abs(expected - predicted)/expected for each of
        # bg_param, bth_param, and pe_param, where expected is the true value of the
        # parameter and predicted is the value returned by the NN model.
        bg_error, bth_error, pe_error = error_ratios.cpu()

        # Save a checkpoint in `checkpoint_dir`/checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(
            loss=loss, bg_error=bg_error, bth_error=bth_error, pe_error=pe_error
        )
        # note: above line utilizes config implicitly
        # TODO: Ask Marissa about other hyperparameter tuning packages that allows
        # dataclasses instead of Ray


def main() -> None:

    project_path = Path("/proj/avdlab/projects/Solutions_ML/")
    log_path = project_path / "mike_outputs/"
    ray_path = log_path / "ray_results/"

    # TODO: read config from a yaml file
    config = data.NNConfig(
        "theoretical_nn_training/configurations/sample_config_rcs_128.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The scheduler chooses when to kill a run early. It does this by comparing the
    # metric (loss) to other runs, and if it is doing relatively poorly according to the
    # mode (min), it will kill it, but only if a grace_period number of epochs (100)
    # has elapsed.
    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=1)

    reporter = CLIReporter(
        metric_columns=["bg_error", "bth_error", "pe_error", "loss"],
        max_report_frequency=60,
        max_progress_rows=50,
    )

    result = tune.run(
        partial(run, my_config=config, device=device),
        config=config.__dict__,
        resources_per_trial={"gpu": 1},  # each run uses this many resources
        num_samples=4,
        scheduler=scheduler,
        keep_checkpoints_num=5,
        checkpoint_score_attr="loss",
        progress_reporter=reporter,
        local_dir=str(ray_path),
    )

    best_trial = result.get_best_trial("loss", "min")
    if best_trial is None:
        raise AllFailedError(result)

    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial config: {best_trial.config}")
    print(f'Best trial final loss: {best_trial.last_result["loss"]:.5f}')

    print(
        result.dataframe().sort_values(by="loss")[
            [
                "loss",
                "bg_error",
                "bth_error",
                "layer_sizes",
                "kernel_sizes",
                "pool_sizes",
                "channels",
            ]
        ]
    )


if __name__ == "__main__":
    main()
