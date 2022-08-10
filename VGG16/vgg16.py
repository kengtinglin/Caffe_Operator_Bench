from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import datetime

from caffe2.python import core, workspace

def benchmark_vgg16(
        batch_size,
        iterations,
        use_gpu):
    print('Preparing lookup table. ' + str(datetime.datetime.now()))



    # Create data (N, C, H, W)
    input = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    print(f'Input has shape = {input.shape}') 
    net = core.Net("mynet")

    if use_gpu == True:
        gpu_device_id = 1
        print("Using GPU...")
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, gpu_device_id)):
            workspace.FeedBlob("X", input)
            workspace.FeedBlob("kernel_1", np.random.randn(64,3,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_2", np.random.randn(64,64,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_3", np.random.randn(128,64,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_4", np.random.randn(128,128,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_5", np.random.randn(256,128,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_6", np.random.randn(256,256,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_7", np.random.randn(512,256,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_8", np.random.randn(512,512,3,3).astype(np.float32))
            workspace.FeedBlob("kernel_9", np.random.rand(4096, 25088).astype(np.float32))
            workspace.FeedBlob("kernel_10", np.random.rand(4096, 4096).astype(np.float32))
            workspace.FeedBlob("kernel_11", np.random.rand(1000, 4096).astype(np.float32))
            
            workspace.FeedBlob("bias_1", np.ones(64, dtype=np.float32))
            workspace.FeedBlob("bias_2", np.ones(128, dtype=np.float32))
            workspace.FeedBlob("bias_3", np.ones(256, dtype=np.float32))
            workspace.FeedBlob("bias_4", np.ones(512, dtype=np.float32))
            workspace.FeedBlob("bias_9", np.ones(4096, dtype=np.float32))
            workspace.FeedBlob("bias_10", np.ones(1000, dtype=np.float32))


            net.Conv(["X", "kernel_1", "bias_1"], "Y1", kernel=3, pad=1, stride=1)
            net.Conv(["Y1", "kernel_2", "bias_1"], "Y2_", kernel=3, pad=1, stride=1)
            net.MaxPool(["Y2_"], ["Y2"], kernel=2, stride=2)
            net.Conv(["Y2", "kernel_3", "bias_2"], "Y3", kernel=3, pad=1, stride=1)
            net.Conv(["Y3", "kernel_4", "bias_2"], "Y4_", kernel=3, pad=1, stride=1)
            net.MaxPool(["Y4_"], ["Y4"], kernel=2, stride=2)
            net.Conv(["Y4", "kernel_5", "bias_3"], "Y5", kernel=3, pad=1, stride=1)
            net.Conv(["Y5", "kernel_6", "bias_3"], "Y6", kernel=3, pad=1, stride=1)
            net.Conv(["Y6", "kernel_6", "bias_3"], "Y7_", kernel=3, pad=1, stride=1)
            net.MaxPool(["Y7_"], ["Y7"], kernel=2, stride=2)
            net.Conv(["Y7", "kernel_7", "bias_4"], "Y8", kernel=3, pad=1, stride=1)
            net.Conv(["Y8", "kernel_8", "bias_4"], "Y9", kernel=3, pad=1, stride=1)
            net.Conv(["Y9", "kernel_8", "bias_4"], "Y10_", kernel=3, pad=1, stride=1)
            net.MaxPool(["Y10_"], ["Y10"], kernel=2, stride=2)
            net.Conv(["Y10", "kernel_8", "bias_4"], "Y11", kernel=3, pad=1, stride=1)
            net.Conv(["Y11", "kernel_8", "bias_4"], "Y12", kernel=3, pad=1, stride=1)
            net.Conv(["Y12", "kernel_8", "bias_4"], "Y13_", kernel=3, pad=1, stride=1)
            net.MaxPool(["Y13_"], ["Y13"], kernel=2, stride=2)
            net.Flatten(["Y13"], ["Y14"], axis=1)
            net.FC(["Y14", "kernel_9", "bias_9"], "Y15")
            net.FC(["Y15", "kernel_10", "bias_9"], "Y16")
            net.FC(["Y16", "kernel_11", "bias_10"], "Y17")

    else:
        workspace.FeedBlob("X", input)
        workspace.FeedBlob("kernel_1", np.random.randn(64,3,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_2", np.random.randn(64,64,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_3", np.random.randn(128,64,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_4", np.random.randn(128,128,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_5", np.random.randn(256,128,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_6", np.random.randn(256,256,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_7", np.random.randn(512,256,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_8", np.random.randn(512,512,3,3).astype(np.float32))
        workspace.FeedBlob("kernel_9", np.random.rand(4096, 25088).astype(np.float32))
        workspace.FeedBlob("kernel_10", np.random.rand(4096, 4096).astype(np.float32))
        workspace.FeedBlob("kernel_11", np.random.rand(1000, 4096).astype(np.float32))
        
        workspace.FeedBlob("bias_1", np.ones(64, dtype=np.float32))
        workspace.FeedBlob("bias_2", np.ones(128, dtype=np.float32))
        workspace.FeedBlob("bias_3", np.ones(256, dtype=np.float32))
        workspace.FeedBlob("bias_4", np.ones(512, dtype=np.float32))
        workspace.FeedBlob("bias_9", np.ones(4096, dtype=np.float32))
        workspace.FeedBlob("bias_10", np.ones(1000, dtype=np.float32))


        net.Conv(["X", "kernel_1", "bias_1"], "Y1", kernel=3, pad=1, stride=1)
        net.Conv(["Y1", "kernel_2", "bias_1"], "Y2_", kernel=3, pad=1, stride=1)
        net.MaxPool(["Y2_"], ["Y2"], kernel=2, stride=2)
        net.Conv(["Y2", "kernel_3", "bias_2"], "Y3", kernel=3, pad=1, stride=1)
        net.Conv(["Y3", "kernel_4", "bias_2"], "Y4_", kernel=3, pad=1, stride=1)
        net.MaxPool(["Y4_"], ["Y4"], kernel=2, stride=2)
        net.Conv(["Y4", "kernel_5", "bias_3"], "Y5", kernel=3, pad=1, stride=1)
        net.Conv(["Y5", "kernel_6", "bias_3"], "Y6", kernel=3, pad=1, stride=1)
        net.Conv(["Y6", "kernel_6", "bias_3"], "Y7_", kernel=3, pad=1, stride=1)
        net.MaxPool(["Y7_"], ["Y7"], kernel=2, stride=2)
        net.Conv(["Y7", "kernel_7", "bias_4"], "Y8", kernel=3, pad=1, stride=1)
        net.Conv(["Y8", "kernel_8", "bias_4"], "Y9", kernel=3, pad=1, stride=1)
        net.Conv(["Y9", "kernel_8", "bias_4"], "Y10_", kernel=3, pad=1, stride=1)
        net.MaxPool(["Y10_"], ["Y10"], kernel=2, stride=2)
        net.Conv(["Y10", "kernel_8", "bias_4"], "Y11", kernel=3, pad=1, stride=1)
        net.Conv(["Y11", "kernel_8", "bias_4"], "Y12", kernel=3, pad=1, stride=1)
        net.Conv(["Y12", "kernel_8", "bias_4"], "Y13_", kernel=3, pad=1, stride=1)
        net.MaxPool(["Y13_"], ["Y13"], kernel=2, stride=2)
        net.Flatten(["Y13"], ["Y14"], axis=1)
        net.FC(["Y14", "kernel_9", "bias_9"], "Y15")
        net.FC(["Y15", "kernel_10", "bias_9"], "Y16")
        net.FC(["Y16", "kernel_11", "bias_10"], "Y17")
    
    workspace.CreateNet(net)
    print('Preparation finished. ' + str(datetime.datetime.now()))

    workspace.BenchmarkNet(net.Name(), 1, iterations, True)
    print(f'Output has shape = {workspace.FetchBlob("Y17").shape}')    


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
    benchmark_vgg16(
        args.batch_size,
        args.iteration,
        args.use_gpu)
