import math
import torch
import torch.nn as nn
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    :param previous_conv: a tensor vector of previous convolution layer
    :param num_sample: an int number of image in the batch
    :param previous_conv_size: an int vector [height, weight] of the matrix features size of previous conv layer
    :param out_pool_size: a int vector of expected output size of max pooling layer
    :return:
    '''
    for i in range(len(out_pool_size)):
        h_wid = (math.ceil(previous_conv_size[0] / out_pool_size[i])) ##向上取整
        w_wid = (math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_std = math.floor(previous_conv_size[0] / out_pool_size[i])
        w_std = math.floor(previous_conv_size[1] / out_pool_size[i])
        h_pad = math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        max_pool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_std, w_std))
        x = max_pool(previous_conv)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp
