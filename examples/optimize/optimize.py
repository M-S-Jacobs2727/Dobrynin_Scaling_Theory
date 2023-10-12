import argparse
import optuna

import torch

from ...core.configuration import Configuration
from ...core.surface_generator import SurfaceGenerator
from ...models.Inception3 import Inception3


def parse_args() -> Configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to a YAML or JSON configuration file",
    )
    parser.add_argument(
        "-l",
        "--logfile",
        type=str,
        default=None,
        help="Path to a log file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional information",
    )

    args = parser.parse_args()

    config: Configuration = Configuration.load(args.config_file)
    return config


def objective(trial: optuna.trial.Trial) -> float:
    resolution = (224, 224)
    train_size = 51200
    batch_size = 64
    loss_fn = torch.nn.MSELoss()
    model = Inception3().to(DEVICE)

    # generate optimizer

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 0.9, log=False)
    beta_2 = trial.suggest_float("beta_2", 0.75, 0.999, log=False)
    eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps)

    print(f'Epoch\tBg_Error\tTime', flush=True)

    for epoch in range(20):
        train_loss, train_pred, train_y = train(
                model, loss_fn, optimizer, DEVICE,
                train_size, batch_size, resolution
                )

        bg_train_true = mike.unnormalize_Bg_param(train_y)
        bg_train_pred = mike.unnormalize_Bg_param(train_pred)

        bg_train_error = torch.mean(torch.abs(bg_train_true-bg_train_pred)/bg_train_true).item()

        trial.report(bg_train_error, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    return bg_train_error


def run(
    config: Configuration,
    generator: SurfaceGenerator,
    model: Inception3,
    optimizer: torch.optim.Optimizer,
) -> None:
    study = optuna.create_study(
        direction="minimize",
        study_name="Bg_Hyperparam_Tune",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=9, interval_steps=2, n_min_trials=5
        ),
    )
    study.optimize(objective, n_trials=50)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("\tValue: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")


def main():
    config = parse_args()

    generator = SurfaceGenerator(config)

    model_bg = Inception3()
    model_bth = Inception3()

    optimizer_bg = torch.optim.Adam(
        model_bg.parameters(),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.epsilon,
        weight_decay=config.optimizer.weight_decay,
    )
    optimizer_bth = torch.optim.Adam(
        model_bth.parameters(),
        lr=config.adam_bth.learning_rate,
        betas=config.adam_bth.betas,
        eps=config.adam_bth.epsilon,
        weight_decay=config.adam_bth.weight_decay,
    )

    run(config, generator, model_bg, optimizer_bg, model_bth, optimizer_bth)


if __name__ == "__main__":
    main()
