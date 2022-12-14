import numpy as np
import math

def get_intensity(object):
    if (object['dtype']) == 'float':
        dbyte = 4

    if (object['operator']) == 'Conv':
        FLOPs = 2 * object['batch_size'] * object['input_channel'] * object['output_height'] * object['output_width'] \
              * object['kernel_num'] * object['kernel_height'] * object['kernel_width']
        Write = object['batch_size'] * object['output_height'] * object['output_width'] * object['output_channel']
        Param = object['kernel_num'] * object['kernel_height'] * object['kernel_width'] * object['kernel_channel']
        Read = object['batch_size'] * object['input_width'] * object['input_height'] * object['input_channel'] + Param
    elif (object['operator']) == 'FC':
        FLOPs = 2 * object['batch_size'] * object['input_shape'] * object['output_shape'] 
        Write = object['batch_size'] * object['output_shape']
        Param = object['input_shape'] * object['output_shape'] + object['output_shape']
        Read = object['batch_size'] * object['input_shape'] + Param
    elif (object['operator']) == 'SLS':
        FLOPs = object['batch_size'] * object['indices_per_lookup'] * object['embedding_col']
        Read = object['batch_size'] * object['indices_per_lookup'] * object['embedding_col']
        Write = object['batch_size'] * object['embedding_col']
        Param = object['embedding_row'] * object['embedding_col']
    intensity = round(FLOPs / ((Read + Write)*dbyte), 2)
    return intensity
