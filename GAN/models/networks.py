import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')


# Resnet Generator

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer.func == nn.InstanceNorm2d if isinstance(norm_layer, functools.partial) else norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        for i in range(2):
            mult = 2 ** i
            model.extend([nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)])

        mult = 2 ** 2
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))

        for i in range(2, 0, -1):
            mult = 2 ** i
            model.extend([nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)])

        model.extend([nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()])
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad2d(1))
        else:
            conv_block.append(nn.ReplicationPad2d(1))

        conv_block.extend([nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)])
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad2d(1))
        else:
            conv_block.append(nn.ReplicationPad2d(1))

        conv_block.extend([nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias), norm_layer(dim)])
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
