"""A simple script to test the speed of surface generation as a function of batch
size and resolution.
"""
import time

import theoretical_nn_training.generators as generators
from theoretical_nn_training.configuration import NNConfig


def main() -> None:

    # TODO: the same for the voxel_image_generator
    print("Begin\n")
    num_samples = 40_960_000

    batches = [8 * 2**i for i in range(10)]
    resolutions = [16, 32, 64, 128]
    batch_time = [[1.0 for _ in batches] for _ in resolutions]
    config = NNConfig("theoretical_nn_training/configurations/sample_config.yaml")

    start = time.perf_counter()
    for i, res in enumerate(resolutions):
        print(f"{res = }")
        for j, batch_size in enumerate(batches):
            config.batch_size = batch_size
            config.phi_range.resolution = res
            config.nw_range.resolution = res
            config.eta_sp_range.resolution = res
            print(f"{batch_size = }")
            num_batches = num_samples // batch_size
            start_batch = time.perf_counter()
            generator = generators.VoxelImageGenerator(config)
            for surf, feat in generator(num_batches):
                pass
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
