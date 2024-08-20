import torch
import torch.nn as nn
from torchvision.models import inception_resnet_v2
from torchsummary import summary

class PyramidFeatureHead(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, features):
        features = nn.functional.relu(self.conv1(features), inplace=True)
        features = nn.functional.relu(self.conv2(features), inplace=True)
        return features

class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, normalizer):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            normalizer(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        return self.sequence(features)

class InceptionFPN(nn.Module):
    def __init__(self, normalizer, output_channels=3, filter_count=128, pyramid_filters=256):
        super().__init__()
        self.fpn = PyramidNetwork(pyramid_filters, normalizer)

        self.segment_heads = [
            PyramidFeatureHead(pyramid_filters, filter_count, filter_count) for _ in range(4)
        ]

        self.smoothing_layers = nn.Sequential(
            nn.Conv2d(4 * filter_count, filter_count, kernel_size=3, padding=1),
            normalizer(filter_count),
            nn.ReLU(),
            nn.Conv2d(filter_count, filter_count // 2, kernel_size=3, padding=1),
            normalizer(filter_count // 2),
            nn.ReLU(),
            nn.Conv2d(filter_count // 2, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        features = self.fpn(inputs)
        features = [seg_head(f) for seg_head, f in zip(self.segment_heads, features)]
        features = torch.cat(features, dim=1)
        features = self.smoothing_layers(features)
        return torch.tanh(features)

class PyramidNetwork(nn.Module):
    def __init__(self, num_filters, normalizer):
        super().__init__()
        self.inception_net = inception_resnet_v2(pretrained=True)
        self.extractors = nn.ModuleList([
            self.inception_net.conv2d_1a,
            nn.Sequential(self.inception_net.conv2d_2a, self.inception_net.conv2d_2b, self.inception_net.maxpool_3a),
            # Additional layers follow the same pattern
        ])
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False),
            # Additional layers should be added for each feature extractor
        ])

    def forward(self, x):
        x = self.extractors[0](x)
        lateral_features = [self.lateral_convs[0](x)]
        for extractor, lateral_conv in zip(self.extractors[1:], self.lateral_convs[1:]):
            x = extractor(x)
            lateral_features.append(lateral_conv(x))
        # Combine lateral features and apply top-down pathway
        return lateral_features

