import torch
import torch.nn as nn
from models.mobilenet_v2 import MobileNetV2
from torch.nn.functional import relu, upsample

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        return relu(self.conv(x), inplace=True)

class FPNMobileNet(nn.Module):
    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()
        net = MobileNetV2(n_class=1000, pretrained=pretrained)
        self.features = net.features

        self.encoders = nn.ModuleList([
            nn.Sequential(*self.features[i:j]) for i, j in zip([0, 2, 4, 7, 11], [2, 4, 7, 11, 16])
        ])

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, num_filters, kernel_size=1, bias=False) 
            for ch in [16, 24, 32, 64, 160]
        ])

        self.top_down_layers = nn.ModuleList([
            ConvRelu(num_filters, num_filters) for _ in range(3)
        ])

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
            nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = [encoder(x) for encoder in self.encoders]
        laterals = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features)]

        for i in range(len(self.top_down_layers) - 1, 0, -1):
            laterals[i - 1] += upsample(laterals[i], scale_factor=2, mode="nearest")

        smoothed = self.smooth(torch.cat(laterals, dim=1))
        final_output = torch.tanh(upsample(smoothed, scale_factor=2, mode="nearest") + x)
        return torch.clamp(final_output, min=-1, max=1)

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

