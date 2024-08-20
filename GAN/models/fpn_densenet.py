import torch
import torch.nn as nn
from torchvision.models import densenet121

class FPNSegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x), inplace=True)
        x = nn.functional.relu(self.conv2(x), inplace=True)
        return x

class FPNDense(nn.Module):
    def __init__(self, output_channels=3, num_filters=128, fpn_filters=256, pretrained=True):
        super().__init__()
        self.fpn = FPN(fpn_filters, pretrained)
        self.heads = nn.ModuleList([FPNSegHead(fpn_filters, num_filters, num_filters) for _ in range(4)])

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters // 2),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(num_filters // 2, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        maps = self.fpn(x)
        for i, head in enumerate(self.heads):
            maps[i + 1] = nn.functional.upsample(head(maps[i + 1]), scale_factor=2**(i + 1), mode="nearest")

        x = self.smooth(torch.cat(maps[1:], dim=1))
        x = nn.functional.upsample(x, scale_factor=2, mode="nearest")
        x = self.smooth2(x + maps[0])
        x = nn.functional.upsample(x, scale_factor=2, mode="nearest")
        return torch.tanh(self.final(x))

    def unfreeze(self):
        for param in self.fpn.parameters():
            param.requires_grad = True

class FPN(nn.Module):
    def __init__(self, num_filters=256, pretrained=True):
        super().__init__()
        features = densenet121(pretrained=pretrained).features
        self.enc0 = nn.Sequential(features.conv0, features.norm0, features.relu0)
        self.pool0 = features.pool0
        self.enc1 = features.denseblock1
        self.enc2 = features.denseblock2
        self.enc3 = features.denseblock3
        self.enc4 = features.denseblock4
        self.norm = features.norm5

        self.trans = [features.transition1, features.transition2, features.transition3]

        self.laterals = nn.ModuleList([
            nn.Conv2d(1024, num_filters, kernel_size=1, bias=False),
            nn.Conv2d(1024, num_filters, kernel_size=1, bias=False),
            nn.Conv2d(512, num_filters, kernel_size=1, bias=False),
            nn.Conv2d(256, num_filters, kernel_size=1, bias=False),
            nn.Conv2d(64, num_filters // 2, kernel_size=1, bias=False),
        ])

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(self.pool0(enc0))
        enc2 = self.enc2(self.trans[0](enc1))
        enc3 = self.enc3(self.trans[1](enc2))
        enc4 = self.enc4(self.trans[2](enc3))
        enc4 = self.norm(enc4)

        maps = [self.laterals[i](enc) for i, enc in enumerate([enc4, enc3, enc2, enc1, enc0])]
        for i in range(1, 5):
            maps[i] += nn.functional.upsample(maps[i - 1], scale_factor=2, mode="nearest")

        return maps
