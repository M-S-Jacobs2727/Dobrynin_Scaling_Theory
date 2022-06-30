import argparse
from pathlib import Path

import pandas
import theoretical_nn_training.data_processing as data
import torch
from theoretical_nn_training.configuration import NNConfig
from theoretical_nn_training.generators import SurfaceGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.is_file():
        raise FileNotFoundError(config_file.absolute())
    output_file = Path(args.output)
    if not output_file.parent.is_dir():
        raise FileNotFoundError(output_file.parent.absolute())

    config = NNConfig(config_file)
    config.batch_size = 1
    gen = SurfaceGenerator(config)
    surf, feat = next(iter(gen(1)))

    eta_sp = data.unnormalize_eta_sp(surf.ravel(), config.eta_sp_range)
    phi_nw_mesh = torch.stack(
        torch.meshgrid(gen.phi.ravel(), gen.Nw.ravel(), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    mask = eta_sp != 1
    df = pandas.DataFrame(
        torch.cat((phi_nw_mesh, eta_sp.reshape(-1, 1)), dim=1)[mask].cpu().numpy(),
        columns=["phi", "Nw", "eta_sp"],
    )
    df.sort_values(by=["Nw", "phi"], inplace=True)
    df.to_csv(output_file, index=False)

    range_list = [config.bg_range, config.bth_range, config.pe_range]
    if config.mode is data.Mode.THETA:
        range_list.pop(0)
    elif config.mode is data.Mode.GOOD:
        range_list.pop(1)
    for f, r in zip(feat.ravel(), range_list):
        print(f"{data.unnormalize_feature(f, r).item():.4f}")


if __name__ == "__main__":
    main()
