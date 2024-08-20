import torch.nn as nn

from .common import ResidualBlock, standard_convolution

def feature_encoder(in_channels, feature_levels):
    """ Feature encoder with progressive downsampling. """
    layers = [
        nn.Conv2d(in_channels, feature_levels, 5, stride=1, padding=2),
        nn.Conv2d(feature_levels, feature_levels * 2, 5, stride=2, padding=2),
        nn.Conv2d(feature_levels * 2, feature_levels * 3, 5, stride=2, padding=2),
    ]
    return nn.Sequential(*layers)

def feature_decoder(output_channels, feature_levels):
    """ Feature decoder with progressive upsampling. """
    upsample_args = {'stride': 2, 'padding': 1, 'output_padding': 1}
    layers = [
        nn.ConvTranspose2d(feature_levels * 3, feature_levels * 2, 3, **upsample_args),
        nn.ConvTranspose2d(feature_levels * 2, feature_levels, 3, **upsample_args),
        nn.Conv2d(feature_levels, output_channels, 5, stride=1, padding=2),
    ]
    return nn.Sequential(*layers)

def deep_residual_network(feature_levels, kernel_size, num_blocks, input_channels=None, output_channels=None):
    """ Deep residual network construction. """
    modules = []

    if input_channels is not None:
        modules.append(standard_convolution(input_channels, feature_levels, kernel_size))

    modules.extend([ResidualBlock(feature_levels, kernel_size) for _ in range(num_blocks)])

    if output_channels is not None:
        modules.append(standard_convolution(feature_levels, output_channels, kernel_size))

    return nn.Sequential(*modules)
