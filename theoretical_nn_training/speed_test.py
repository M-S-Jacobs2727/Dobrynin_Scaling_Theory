import time

import torch

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as generators


def main() -> None:
    device = torch.device("cuda")

    # TODO: the same for the voxel_image_generator
    print("Begin\n")
    num_samples = 40_960_000

    batches = [8 * 2**i for i in range(10)]
    resolutions = [16, 32, 64, 128]
    batch_time = [[1.0 for _ in batches] for _ in resolutions]
    config = data.NNConfig("theoretical_nn_training/configurations/sample_config.yaml")

    start = time.perf_counter()
    for i, res in enumerate(resolutions):
        print(f"{res = }")
        for j, batch_size in enumerate(batches):
            config.batch_size = batch_size
            config.resolution = data.Resolution(res, res, res)
            print(f"{batch_size = }")
            num_batches = num_samples // batch_size
            start_batch = time.perf_counter()
            for surf in generators.voxel_image_generator(num_batches, device, config):
                X, y = surf
            elapsed = time.perf_counter() - start_batch
            batch_time[i][j] = elapsed

    total = time.perf_counter() - start

    print(f"{batches = }\n{resolutions = }")
    print(f"Total time = {total}s")
    print()
    print("  batch_size  |  resolution  |  time (s)  |  rate (samples/sec)")
    print("=================================================================")
    for r, time_list in zip(resolutions, batch_time):
        for bs, t in zip(batches, time_list):
            print(
                f"  {bs:10d}  |  ({r:3d}, {r:3d})  |  {t:.2e}  |"
                f"  {num_samples/t:.1f}"
            )


if __name__ == "__main__":
    main()
