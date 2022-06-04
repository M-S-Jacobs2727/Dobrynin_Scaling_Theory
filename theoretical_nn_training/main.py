from pathlib import Path
from typing import Any, Dict

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import theoretical_nn_training.generators as gen
import theoretical_nn_training.loss_funcs as loss
import theoretical_nn_training.models as models
import theoretical_nn_training.training as train
from theoretical_nn_training.datatypes import Resolution


def run(config: Dict[str, Any], checkpoint_dir: str) -> None:

    if config["resolution"].eta_sp:
        generator = gen.voxel_image_generator
    else:
        generator = gen.surface_generator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.ConvNeuralNet3D(
        res=config["resolution"], l1=config["l1"], l2=config["l2"]
    ).to(device)

    loss_fn = config["loss_fn"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optim_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    for epoch in range(config["epochs"]):

        train.train(
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_samples=config["train_size"],
            batch_size=config["batch_size"],
            resolution=config["resolution"],
            generator=generator,
            optimizer=optimizer,
        )

        (bg_err, bth_err, pe_err), loss = train.test(
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_samples=config["test_size"],
            batch_size=config["batch_size"],
            resolution=config["resolution"],
            generator=generator,
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

    config = {
        "batch_size": 100,
        "train_size": 70000,
        "test_size": 2000,
        "resolution": Resolution(256, 256),
        "loss_fn": loss.CustomMSELoss(),
        "lr": 1e-3,
    }

    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=100)
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "l3"],
        metric_columns=["bg_err", "bth_err", "pe_err", "loss"],
        max_report_frequency=60,
        max_progress_rows=50,
    )

    result = tune.run(
        run,
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=50,
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
