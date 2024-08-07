import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, down_sampling=None):
        super().__init__()
        self.down_sampling = down_sampling
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = self.relu(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))

        if self.down_sampling is not None:
            identity = self.down_sampling(identity)

        x += identity
        x = self.relu(x)

        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sampling=None):
        super().__init__()
        self.down_sampling = down_sampling
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.conv_3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_3 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x):
        identity = x

        x = self.relu(self.norm_1(self.conv_1(x)))
        x = self.relu(self.norm_2(self.conv_2(x)))
        x = self.norm_3(self.conv_3(x))

        if self.down_sampling is not None:
            identity = self.down_sampling(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=1000, in_channels=3, channel_ratio=1.0):
        super().__init__()
        self.in_channels = make_divisible(64 * channel_ratio)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, make_divisible(32 * channel_ratio),
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(make_divisible(32 * channel_ratio)),
            nn.ReLU(inplace=True),
            nn.Conv2d(make_divisible(32 * channel_ratio), make_divisible(32 * channel_ratio),
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(make_divisible(32 * channel_ratio)),
            nn.ReLU(inplace=True),
            nn.Conv2d(make_divisible(32 * channel_ratio), make_divisible(64 * channel_ratio),
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(make_divisible(64 * channel_ratio)),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, make_divisible(64 * channel_ratio), layers[0])
        )
        self.conv_3 = self._make_layer(block, make_divisible(128 * channel_ratio), layers[1], stride=2)
        self.conv_4 = self._make_layer(block, make_divisible(256 * channel_ratio), layers[2], stride=2)
        self.conv_5 = self._make_layer(block, make_divisible(512 * channel_ratio), layers[3], stride=2)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(make_divisible(512 * channel_ratio) * block.expansion, classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        down_sampling = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sampling = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True),
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, down_sampling)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet_18(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


def resnet_34(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


def resnet_50(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BottleneckBlock, [3, 4, 6, 3],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


def resnet_101(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BottleneckBlock, [3, 4, 23, 3],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


def resnet_152(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BottleneckBlock, [3, 8, 36, 3],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


def resnet_10(classes=1000, in_channels=3, channel_ratio=1.0):
    return ResNet(BasicBlock, [1, 1, 1, 1],
                  classes=classes, in_channels=in_channels, channel_ratio=channel_ratio)


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = resnet_18(classes=1000, in_channels=3, channel_ratio=1.0)
    out_data = model(in_data)
