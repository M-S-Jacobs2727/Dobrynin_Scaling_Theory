import sys

import torch

import psst
from psst.models import Inception3


def main():
    device = torch.device("cpu")

    config = psst.loadConfig(sys.argv[1])
    run_config = config.run_config
    adam_config = config.adam_config
    generator_config = config.generator_config

    checkpoint_file = None if len(sys.argv) < 3 else sys.argv[2]

    model = Inception3()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **adam_config)
    start_epoch = 0
    if checkpoint_file:
        chkpt: psst.Checkpoint = torch.load(checkpoint_file)
        start_epoch = chkpt.epoch
        model.load_state_dict(chkpt.model_state)
        optimizer.load_state_dict(chkpt.optimizer_state)

    loss_fn = torch.nn.MSELoss()
    generator = psst.SampleGenerator(**generator_config, device=device)

    psst.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        generator=generator,
        start_epoch=start_epoch,
        num_epochs=run_config.num_epochs,
        num_samples_train=run_config.num_samples_train,
        num_samples_test=run_config.num_samples_test,
        checkpoint_filename=run_config.checkpoint_filename,
        checkpoint_frequency=-1,
    )

    torch.save(
        psst.Checkpoint(
            run_config.num_epochs,
            model.state_dict(),
            optimizer.state_dict(),
        ),
        "train_bg_final_state.pt",
    )


if __name__ == "__main__":
    main()
