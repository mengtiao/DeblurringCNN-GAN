import torch.nn as nn
import math

def conv_bn_relu(inp, oup, kernel_size, stride, padding, groups=1):
    """ Combine convolutions and activation layers. """
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (stride == 1 and inp == oup)

        hidden_dim = round(inp * expand_ratio)
        layers = []
        if expand_ratio != 1:
            # expand input features
            layers.append(conv_bn_relu(inp, hidden_dim, 1, 1, 0))
        # depthwise convolution
        layers.extend([
            conv_bn_relu(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        last_channel = 1280
        config = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer: regular conv.
        input_channel = int(32 * width_mult)
        features = [conv_bn_relu(3, input_channel, 3, 2, 1)]
        # Inverted residual blocks
        for t, c, n, s in config:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Last layers
        features.extend([
            conv_bn_relu(input_channel, int(last_channel * width_mult), 1, 1, 0),
            nn.AdaptiveAvgPool2d(1)
        ])
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(last_channel * width_mult), n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
