"""A simple script to test the speed of surface generation as a function of batch
size and resolution.
"""
import time

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as generators
from theoretical_nn_training.configuration import NNConfig


def main() -> None:

    # TODO: the same for the voxel_image_generator
    print("Begin\n")
    num_samples = 40_960_000

    batches = [32, 64, 128]
    resolutions = [64, 128, 256, 512]
    config = NNConfig("theoretical_nn_training/configurations/mixed_config_512.yaml")

    print(f"{batches = }\n{resolutions = }")
    print()
    print("  batch_size  |  resolution  |  time (s)  |  rate (samples/sec)")
    print("=================================================================")

    for res in resolutions:
        config.resolution = data.Resolution(res, res)
        generator = generators.SurfaceGenerator(config)
        for batch_size in batches:
            generator.config.batch_size = batch_size
            num_batches = num_samples // batch_size
            start_batch = time.perf_counter()
            for surf, feat in generator(num_batches):
                pass
            elapsed = time.perf_counter() - start_batch
            print(
                f"  {batch_size:10d}  |  ({res:3d}, {res:3d})  |  {elapsed:.2e}  |"
                f"  {num_samples/elapsed:.1f}"
            )


if __name__ == "__main__":
    main()
