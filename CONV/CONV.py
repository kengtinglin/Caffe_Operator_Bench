from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import datetime

from caffe2.python import core, workspace


def benchmark_CONV(
        batch_size,
        iterations,
        use_gpu):

    # Create data (N, C, H, W)
    data = np.random.randn(batch_size,64,224,224).astype(np.float32)
    print(f'Input has shape = {data.shape}') 
    # Create Filter (M, c, kh, kw)
    filters = np.random.randn(64,64,3,3).astype(np.float32)
    print(f'Kernel has shape = {filters.shape}')
    # Create Bias : (M)
    bias = np.ones([64], dtype=np.float32)
    print(f'Bias has shape = {bias.shape}')

    net = core.Net("mynet")

    if use_gpu == True:
        # Set GPU device
        gpu_device_id = 1
        print("Using GPU...")
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, gpu_device_id)):
            workspace.FeedBlob("X", data)
            workspace.FeedBlob("filter", filters)
            workspace.FeedBlob("bias", bias)
        
            net.Conv(["X", "filter", "bias"], "Y", kernel=3, pad=1, stride=1)
    else:
        workspace.FeedBlob("X", data)
        workspace.FeedBlob("filter", filters)
        workspace.FeedBlob("bias", bias)
    
        net.Conv(["X", "filter", "bias"], "Y", kernel=3, pad=1, stride=1)
    
    workspace.CreateNet(net)
    print('Preparation finished. ' + str(datetime.datetime.now()))

    workspace.BenchmarkNet(net.Name(), 1, iterations, True)
    print(f'Output has shape = {workspace.FetchBlob("Y").shape}')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for convolution.")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="The batch size.")
    parser.add_argument(
        '-i', "--iteration", type=int, default=1,
        help="The number of iterations.")
    parser.add_argument(
        '--use-gpu', action="store_true", default=False,
        help="Use gpu or not.")
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_CONV(
        args.batch_size,
        args.iteration,
        args.use_gpu)
