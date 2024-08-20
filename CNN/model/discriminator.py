import torch
import torch.nn as nn

class GANDiscriminator(nn.Module):
    def __init__(self, config):
        super(GANDiscriminator, self).__init__()
        
        num_features = config.num_features
        kernel_size = config.kernel_size

        def create_conv_layer(kernel_size, in_channels, out_channels, stride, padding=None):
            if padding is None:
                padding = (kernel_size - 1) // 2
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

        self.feature_layers = nn.ModuleList([
            create_conv_layer(kernel_size, 3,         num_features // 2, 1),  # Initial layer
            create_conv_layer(kernel_size, num_features // 2, num_features // 2, 2),  # Downsample to 128
            create_conv_layer(kernel_size, num_features // 2, num_features,   1),
            create_conv_layer(kernel_size, num_features,   num_features,   2),  # Downsample to 64
            create_conv_layer(kernel_size, num_features,   num_features * 2, 1),
            create_conv_layer(kernel_size, num_features * 2, num_features * 2, 4),  # Downsample to 16
            create_conv_layer(kernel_size, num_features * 2, num_features * 4, 1),
            create_conv_layer(kernel_size, num_features * 4, num_features * 4, 4),  # Downsample to 4
            create_conv_layer(kernel_size, num_features * 4, num_features * 8, 1),
            create_conv_layer(4,           num_features * 8, num_features * 8, 4, 0),  # Downsample to 1
        ])

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(num_features * 8, 1, 1, bias=False)

    def forward(self, input):
        for layer in self.feature_layers:
            input = self.activation(layer(input))

        output = self.classifier(input)

        return output
