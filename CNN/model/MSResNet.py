import torch
import torch.nn as nn

from . import common
from .ResNet import ResNet

def initialize_msresnet_model(config):
    return MultiScaleResNet(config)

class FinalConvLayer(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, kernel_size=5, upsample_scale=2):
        super(FinalConvLayer, self).__init__()
        layers = [
            common.default_conv(input_channels, output_channels, kernel_size),
            nn.PixelShuffle(upsample_scale)
        ]
        self.conv_sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_sequence(x)

class MultiScaleResNet(nn.Module):
    def __init__(self, config):
        super(MultiScaleResNet, self).__init__()
        self.rgb_range = config.rgb_range
        self.midpoint = self.rgb_range / 2

        self.num_res_blocks = config.n_resblocks
        self.feature_size = config.n_feats
        self.kernel_size = config.kernel_size

        self.num_scales = config.n_scales

        self.resnet_blocks = nn.ModuleList([
            ResNet(config, 3, 3, mean_shift=False),
        ])
        for _ in range(1, self.num_scales):
            self.resnet_blocks.insert(0, ResNet(config, 6, 3, mean_shift=False))

        self.final_layers = nn.ModuleList([None])
        for _ in range(1, self.num_scales):
            self.final_layers += [FinalConvLayer(3, 12)]

    def forward(self, image_pyramid):
        descending_scales = range(self.num_scales-1, -1, -1)

        for scale in descending_scales:
            image_pyramid[scale] -= self.midpoint

        transformed_pyramid = [None] * self.num_scales

        transformed_input = image_pyramid[-1]
        for scale in descending_scales:
            transformed_pyramid[scale] = self.resnet_blocks[scale](transformed_input)
            if scale > 0:
                upscaled_features = self.final_layers[scale](transformed_pyramid[scale])
                transformed_input = torch.cat((image_pyramid[scale-1], upscaled_features), 1)

        for scale in descending_scales:
            transformed_pyramid[scale] += self.midpoint

        return transformed_pyramid
