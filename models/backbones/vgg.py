import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class VGG(nn.Module):
    def __init__(self, layers, classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0):
        super().__init__()
        self.layer_1 = self._make_layer(
            in_channels, make_divisible(64 * channel_ratio), layers[0])
        self.layer_2 = self._make_layer(
            make_divisible(64 * channel_ratio), make_divisible(128 * channel_ratio), layers[1])
        self.layer_3 = self._make_layer(
            make_divisible(128 * channel_ratio), make_divisible(256 * channel_ratio), layers[2])
        self.layer_4 = self._make_layer(
            make_divisible(256 * channel_ratio), make_divisible(512 * channel_ratio), layers[3])
        self.layer_5 = self._make_layer(
            make_divisible(512 * channel_ratio), make_divisible(512 * channel_ratio), layers[4])

        self.avg_pooling = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Sequential(
            nn.Linear(make_divisible(512 * channel_ratio) * 7 * 7, make_divisible(4096 * channel_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(make_divisible(4096 * channel_ratio), make_divisible(4096 * channel_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(make_divisible(4096 * channel_ratio), make_divisible(classes * channel_ratio))
        )

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks):
        layers = []
        for i in range(0, blocks):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def vgg11(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0):
    return VGG([1, 1, 2, 2, 2],
               classes=classes, in_channels=in_channels, dropout=dropout, channel_ratio=channel_ratio)


def vgg13(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0):
    return VGG([2, 2, 2, 2, 2],
               classes=classes, in_channels=in_channels, dropout=dropout, channel_ratio=channel_ratio)


def vgg16(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0):
    return VGG([2, 2, 3, 3, 3],
               classes=classes, in_channels=in_channels, dropout=dropout, channel_ratio=channel_ratio)


def vgg19(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0):
    return VGG([2, 2, 4, 4, 4],
               classes=classes, in_channels=in_channels, dropout=dropout, channel_ratio=channel_ratio)


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = vgg16(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0)
    out_data = model(in_data)
