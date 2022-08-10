from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import datetime

from caffe2.python import core, workspace

DTYPES = {
    'uint8': np.uint8,
    'uint8_fused': np.uint8,
    'float': np.float32,
    'float16': np.float16,
}


def benchmark_fc(
        batch_size,
        iterations,
        use_gpu):
    print('Preparing lookup table. ' + str(datetime.datetime.now()))

    # Create data (M, k)
    data = np.random.rand(batch_size, 25088).astype(np.float32)
    print(f'Input has shape = {data.shape}') 
    # Create W (N, k)
    weights_1 = np.random.rand(4096, 25088).astype(np.float32)
    # weights_2 = np.random.rand(32, 64).astype(np.float32)
    # weights_3 = np.random.rand(1, 64).astype(np.float32)
    # weights = weights[np.newaxis,:]
    # print(f'Weight has shape = {weights.shape}')
    # Create Bias : (N)
    bias_1 = np.ones(4096, dtype=np.float32)
    # bias_2 = np.ones(32, dtype=np.float32)
    # bias_3 = np.ones(1, dtype=np.float32)
    # bias = np.array([1.,1.,1.]).astype(np.float32)
    # bias = np.array([1.]).astype(np.float32)
    # data = np.ones([categorical_limit, embedding_size], dtype=np.float32)
    # print(f'Bias has shape = {bias.shape}')

    net = core.Net("mynet")

    if use_gpu == True:
        # set GPU device
        gpu_device_id = 1
        print("Using GPU...")
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, gpu_device_id)):
            workspace.FeedBlob("X", data)
            workspace.FeedBlob("W_1", weights_1)
            workspace.FeedBlob("b_1", bias_1)
            # workspace.FeedBlob("W_2", weights_2)
            # workspace.FeedBlob("b_2", bias_2)
            # workspace.FeedBlob("W_3", weights_3)
            # workspace.FeedBlob("b_3", bias_3)
            
            net.FC(["X", "W_1", "b_1"], "Y_1")
            # net.FC(["Y_1", "W_2", "b_2"], "Y_2")
            # net.FC(["Y_2", "W_3", "b_3"], "Y_3")
    else:
        workspace.FeedBlob("X", data)
        workspace.FeedBlob("W_1", weights_1)
        workspace.FeedBlob("b_1", bias_1)
        # workspace.FeedBlob("W_2", weights_2)
        # workspace.FeedBlob("b_2", bias_2)
        # workspace.FeedBlob("W_3", weights_3)
        # workspace.FeedBlob("b_3", bias_3)
        
        net.FC(["X", "W_1", "b_1"], "Y_1")
        # net.FC(["Y_1", "W_2", "b_2"], "Y_2")
        # net.FC(["Y_2", "W_3", "b_3"], "Y_3")
        
    
    workspace.CreateNet(net)
    print('Preparation finished. ' + str(datetime.datetime.now()))

    iteratioins = 1
    workspace.BenchmarkNet(net.Name(), 1, iterations, True)
    print(f'Output has shape = {workspace.FetchBlob("Y_1")}')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum.")
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
    benchmark_fc(
        args.batch_size,
        args.iteration,
        args.use_gpu)
