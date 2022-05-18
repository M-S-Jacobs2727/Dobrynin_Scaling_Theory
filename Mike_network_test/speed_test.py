import time
import torch

import scaling_torch_lib as scaling


def main():
    device = torch.device('cuda')
    num_samples = 40_960_000

    batches = [8*2**i for i in range(10)]
    resolutions = [16, 32, 64, 128]
    batch_time = [[1 for _ in batches] for _ in resolutions]

    start = time.perf_counter()
    for i, res in enumerate(resolutions):
        print(f'{res = }')
        resolution = (res, res)
        for j, batch_size in enumerate(batches):
            print(f'{batch_size = }')
            num_batches = num_samples // batch_size
            start_batch = time.perf_counter()
            for surf in scaling.surface_generator(
                    num_batches, batch_size, device, resolution):
                X, y = surf
            elapsed = time.perf_counter() - start_batch
            batch_time[i][j] = elapsed

    total = time.perf_counter() - start

    print(f'{batches = }\n{resolutions = }')
    print(f'Total time = {total}s')
    print()
    print('  batch_size  |  resolution  |  time (s)  |  rate (samples/sec)')
    print('=================================================================')
    for r, time_list in zip(resolutions, batch_time):
        for bs, t in zip(batches, time_list):
            print(f'  {bs:10d}  |  ({r:3d}, {r:3d})  |  {t:.2e}  |'
                  f'  {num_samples/t:.1f}')
