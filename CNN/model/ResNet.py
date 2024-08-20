import torch.nn as nn
from . import common

def construct_resnet(config):
    return EnhancedResNet(config)

class EnhancedResNet(nn.Module):
    def __init__(self, config, input_channels=3, output_channels=3, features=None, kernel=None, blocks=None, shift_mean=True):
        super(EnhancedResNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.features = config.n_feats if features is None else features
        self.kernel = config.kernel_size if kernel is None else kernel
        self.blocks = config.n_resblocks if blocks is None else blocks

        self.shift_mean = shift_mean
        self.range = config.rgb_range
        self.mean_adjust = self.range / 2

        layers = [
            common.default_conv(self.input_channels, self.features, self.kernel)
        ]
        layers += [common.ResBlock(self.features, self.kernel) for _ in range(self.blocks)]
        layers.append(common.default_conv(self.features, self.output_channels, self.kernel))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.shift_mean:
            x -= self.mean_adjust

        result = self.network(x)

        if self.shift_mean:
            result += self.mean_adjust

        return result
