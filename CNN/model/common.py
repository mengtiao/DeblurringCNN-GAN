import math
import torch
import torch.nn as nn

def create_convolution(in_channels, out_channels, kernel_size, use_bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=use_bias, groups=groups)

def create_normalization(num_features):
    return nn.BatchNorm2d(num_features)

def create_activation():
    return nn.ReLU(True)

def initialize_hidden_state(tensor, num_features):
    batch_size = tensor.size(0)
    height, width = tensor.size()[-2:]
    return tensor.new_zeros((batch_size, num_features, height//4, width//4))

class InputNormalization(nn.Conv2d):
    """Standardize input using a convolution layer."""
    def __init__(self, mean_values=(0.0, 0.0, 0.0), std_devs=(1.0, 1.0, 1.0)):
        super(InputNormalization, self).__init__(3, 3, kernel_size=1)
        mean_tensor = torch.Tensor(mean_values)
        std_tensor = torch.Tensor(std_devs).reciprocal()

        self.weight.data = torch.eye(3).mul(std_tensor).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(-mean_tensor.mul(std_tensor))

        self.weight.requires_grad = False
        self.bias.requires_grad = False

class ConvBlock(nn.Sequential):
    """A block of convolutional layers with optional normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True,
                 conv_layer=create_convolution, norm_layer=None, activation_layer=None):
        modules = [conv_layer(in_channels, out_channels, kernel_size, bias=use_bias)]
        if norm_layer:
            modules.append(norm_layer(out_channels))
        if activation_layer:
            modules.append(activation_layer())
        super(ConvBlock, self).__init__(*modules)

class ResidualBlock(nn.Module):
    def __init__(self, num_features, kernel_size, use_bias=True,
                 conv_layer=create_convolution, norm_layer=None, activation_layer=None):
        super(ResidualBlock, self).__init__()
        modules = [conv_layer(num_features, num_features, kernel_size, bias=use_bias)]
        if norm_layer:
            modules.append(norm_layer(num_features))
        if activation_layer:
            modules.append(activation_layer())

        self.conv_path = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.conv_path(x) + x

class UpscaleBlock(nn.Sequential):
    """Upscale spatial dimensions using convolution and pixel shuffle."""
    def __init__(self, scaling_factor, num_features, use_bias=True,
                 conv_layer=create_convolution, norm_layer=None, activation_layer=None):
        modules = []
        if scaling_factor in [2, 4]:
            for _ in range(int(math.log(scaling_factor, 2))):
                modules.append(conv_layer(num_features, num_features * 4, 3, bias=use_bias))
                modules.append(nn.PixelShuffle(2))
                if norm_layer:
                    modules.append(norm_layer(num_features))
                if activation_layer:
                    modules.append(activation_layer())
        super(UpscaleBlock, self).__init__(*modules)

class DownscaleBlock(nn.Sequential):
    """Downscale spatial dimensions and increase channel dimensions."""
    def __init__(self, scale, num_features, use_bias=True,
                 conv_layer=create_convolution, norm_layer=None, activation_layer=None):
        super(DownscaleBlock, self).__init__()
        if scale == 0.5:
            self.add_module("pixel_sorting", PixelSort(scale=0.5))
            self.add_module("conv", conv_layer(num_features * 4, num_features, 3, bias=use_bias))
            if norm_layer:
                self.add_module("norm", norm_layer(num_features))
            if activation_layer:
                self.add_module("act", activation_layer())

