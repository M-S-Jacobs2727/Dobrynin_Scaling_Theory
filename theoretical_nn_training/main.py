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
import theoretical_nn_training.training as train
from theoretical_nn_training.data_processing import Param, Resolution


def run(config: Dict[str, Any], checkpoint_dir: str, my_config: data.NNConfig) -> None:

    model = models.LinearNeuralNet(
        res=my_config.resolution, layers=my_config.layers
    ).to(my_config.device)

    loss_fn = loss_funcs.CustomMSELoss(
        my_config.bg_param, my_config.bth_param, my_config.pe_param
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config.lr)

    if checkpoint_dir:
        model_state, optim_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    for epoch in range(my_config.epochs):

        train.train(model=model, optimizer=optimizer, loss_fn=loss_fn, config=my_config)

        (bg_err, bth_err, pe_err), loss = train.test(
            model=model, loss_fn=loss_fn, config=my_config
        )
        bg_err = bg_err.cpu()
        bth_err = bth_err.cpu()
        pe_err = pe_err.cpu()

        if (epoch + 1) % 5 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = Path(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss, bg_err=bg_err, bth_err=bth_err, pe_err=pe_err)


def main() -> None:

    proj_path = Path("/proj/avdlab/projects/Solutions_ML/")
    log_path = Path(proj_path, "mike_outputs/")
    ray_path = Path(log_path, "ray_results/")

    config = data.CNNConfig(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        phi_param=Param(1e-6, 1e-2),
        nw_param=Param(100, 1e5),
        eta_sp_param=Param(1, 1e6),
        bg_param=Param(0.3, 1.1),
        bth_param=Param(0.2, 0.8),
        pe_param=Param(4, 20),
        batch_size=100,
        train_size=700_000,
        test_size=2_000,
        epochs=100,
        resolution=Resolution(128, 128),
        lr=0.001,
        layers=[1024, 1024, 512],
        channels=[4, 16, 64, 256],
        kernels=[5, 5, 5, 5],
        pools=[2, 3, 2, 3],
    )

    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=100)
    reporter = CLIReporter(
        parameter_columns=["l1", "l2"],
        metric_columns=["bg_err", "bth_err", "pe_err", "loss"],
        max_report_frequency=60,
        max_progress_rows=50,
    )

    result = tune.run(
        partial(run, my_config=config),
        resources_per_trial={"gpu": 1},
        config=config.__dict__,
        num_samples=5,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=str(ray_path),
    )

    best_trial = result.get_best_trial("loss", "min")
    if best_trial:
        print(f"Best trial: {best_trial.trial_id}")
        print(f"Best trial config: {best_trial.config}")
        print(f'Best trial final loss: {best_trial.last_result["loss"]:.5f}')

    df = result.dataframe()
    print(df.sort_values(by="loss"))


if __name__ == "__main__":
    main()
