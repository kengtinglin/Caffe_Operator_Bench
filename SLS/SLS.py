from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import datetime
import random
import time

from caffe2.python import core, workspace

def benchmark_sparse_lengths_sum(
        categorical_limit,
        embedding_size,
        average_len,
        batch_size,
        iterations,
        lookup_fixed,
        embedding_num,
        use_gpu
        ):
    print('Preparing lookup table. ' + str(datetime.datetime.now()))

    # We will use a constant, but non-trivial value so we save initialization
    # time.
    # categorical_limit = Number of rows of embedding table.
    # embedding_size = Dimensions of embedding table.
    data = []
    for i in range(embedding_num):
        data.append(np.random.rand(categorical_limit, embedding_size).astype(np.float32))
    
    # In order to produce truly random lengths and indices, we will embed a
    # Python operator in the net to generate them.
    def f(_, outputs):
        if lookup_fixed == True:
            lengths = average_len * np.ones(batch_size).astype(np.int32)
        else:
            lengths = np.random.randint(
                int(average_len * 0.75),
                int(average_len * 1.25),
                batch_size).astype(np.int32)
        indices = np.random.randint(
            0, categorical_limit, np.sum(lengths)).astype(np.int64)

        outputs[0].feed(indices)
        outputs[1].feed(lengths)

    net = core.Net("mynet")
    for i in range(embedding_num):
        index = "indices" + str(i)
        lengths = "lengths" + str(i)
        net.Python(f)([], [index, lengths, ])
    if use_gpu == True:
        # Set GPU device
        gpu_device_id = 1   
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, gpu_device_id)):
            for i in range(embedding_num):
                X = "X" + str(i)
                index = "indices" + str(i)
                lengths = "lengths" + str(i)
                Y = "Y" + str(i)
                workspace.FeedBlob(X, data[i].astype(np.float32))
                net.SparseLengthsSum([X, index, lengths], Y)
    else:
        for i in range(embedding_num):
            X = "X" + str(i)
            index = "indices" + str(i)
            lengths = "lengths" + str(i)
            Y = "Y" + str(i)
            workspace.FeedBlob(X, data[i].astype(np.float32))
            a = net.SparseLengthsSum([X, index, lengths], Y)


    workspace.CreateNet(net)
    
    workspace.BenchmarkNet(net.Name(), 1, iterations, True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum.")
    parser.add_argument(
        '-e', "--arch-embedding-size", type=int, default=4000000,
        help="Lookup table size.")
    parser.add_argument(
        "--arch-sparse-feature-size", type=int, default=32,
        help="Embedding dimension.")
    parser.add_argument(
        "--num-indices-per-lookup", type=int, default=80,
        help="Sparse feature average lengths, default is ")
    parser.add_argument(
        "--mini-batch-size", type=int, default=1,
        help="The mini-batch size.")
    parser.add_argument(
        '-i', "--iteration", type=int, default=2,
        help="The number of iterations.")
    parser.add_argument(
        '--lookup-fixed', type=bool, default=True,
        help="The number of per lookup is fixed or not.")
    parser.add_argument(
        '--embedding-num', type=int, default=1,
        help="The number of embedding tables.")
    parser.add_argument(
        '--use-gpu', action="store_true", default=False,
        help="Use gpu or not.")
    
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_sparse_lengths_sum(
        args.arch_embedding_size,
        args.arch_sparse_feature_size,
        args.num_indices_per_lookup,
        args.mini_batch_size,
        args.iteration,
        args.lookup_fixed,
        args.embedding_num,
        args.use_gpu)
    
