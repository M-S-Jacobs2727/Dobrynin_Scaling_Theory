from functools import partial
import logging
import sys
import numpy as np
import torch

from psst.configuration import *
from psst.surface_generator import *
from psst.training import *
from psst.models.Inception3 import Inception3


def run(
        config: Configuration,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        generator: SurfaceGenerator,
        last_epoch: int,
):
    log = logging.getLogger("psst.main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    
    log.info("Starting run of %d epochs", config.epochs)
    for epoch in range(last_epoch, config.epochs):
        log.info("Running epoch %d", epoch)
        avg_loss_train, true_values_train, pred_values_train = train(model, generator, optimizer, loss_fn, config.train_size)
        avg_loss_test, true_values_test, pred_values_test = validate(model, generator, loss_fn, config.test_size)
    
    save_checkpoint(config.checkpoint_file, config.epochs, model, optimizer)


def main():
    device = torch.device("cpu")
    config = getConfig(sys.argv[1])

    model = Inception3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), **config.adam_config.asdict())
    generator = SurfaceGenerator(config.generator_config, device)
    loss_fn = torch.nn.MSELoss()
    last_epoch = 0

    if config.continuing:
        last_epoch = load_checkpoint(config.checkpoint_file, model, optimizer)
    
    run(config, model, optimizer, loss_fn, generator, last_epoch)


if __name__ == '__main__':
    main()
