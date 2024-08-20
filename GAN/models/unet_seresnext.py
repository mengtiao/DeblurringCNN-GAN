import torch
from torch import nn
import torchvision
from torch.nn import functional as F

# Helper functions
def create_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if activation:
        layers.append(activation(inplace=True))
    return nn.Sequential(*layers)

# Building blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_relu = create_conv(in_channels, out_channels, 3, padding=1, activation=nn.ReLU)

    def forward(self, x):
        return self.conv_relu(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upsample_mode='deconv'):
        super(UpConvBlock, self).__init__()
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(middle_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                create_conv(middle_channels, out_channels, 3, padding=1)
            )
        self.conv = create_conv(in_channels, middle_channels, 3, padding=1, activation=nn.ReLU)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

# Main model
class AdvancedUNet(nn.Module):
    def __init__(self, num_classes=3, base_filters=32, use_pretrained=True):
        super(AdvancedUNet, self).__init__()
        self.base_filters = base_filters
        from models.senet import se_resnext50_32x4d
        pretrain_type = 'imagenet' if use_pretrained else None
        self.encoder = se_resnext50_32x4d(pretrained=pretrain_type)

        self.down_layers = nn.ModuleList([
            ConvBlock(3, self.base_filters),
            ConvBlock(self.base_filters, self.base_filters * 2),
            ConvBlock(self.base_filters * 2, self.base_filters * 4)
        ])

        self.up_layers = nn.ModuleList([
            UpConvBlock(self.base_filters * 8, self.base_filters * 8, self.base_filters * 4),
            UpConvBlock(self.base_filters * 4, self.base_filters * 4, self.base_filters * 2),
            UpConvBlock(self.base_filters * 2, self.base_filters * 2, self.base_filters)
        ])

        self.final_conv = nn.Conv2d(self.base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        connections = []
        for down in self.down_layers:
            x = down(x)
            connections.append(x)
        
        for up in reversed(self.up_layers):
            x = up(x)
            x = torch.cat([x, connections.pop()], dim=1)
        
        return self.final_conv(x)

# Using the model
model = AdvancedUNet(num_classes=3)
