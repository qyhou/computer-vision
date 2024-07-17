import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.norm_1(self.conv_1(x)))
        x = self.relu(self.norm_2(self.conv_2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, classes=10, channel_ratio=1.0):
        super().__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_layer_1 = DoubleConv(in_channels, make_divisible(64 * channel_ratio))
        self.left_layer_2 = DoubleConv(make_divisible(64 * channel_ratio), make_divisible(128 * channel_ratio))
        self.left_layer_3 = DoubleConv(make_divisible(128 * channel_ratio), make_divisible(256 * channel_ratio))
        self.left_layer_4 = DoubleConv(make_divisible(256 * channel_ratio), make_divisible(512 * channel_ratio))

        self.middle_layer = DoubleConv(make_divisible(512 * channel_ratio), make_divisible(1024 * channel_ratio))

        self.up_sampling_1 = nn.ConvTranspose2d(
            make_divisible(1024 * channel_ratio), make_divisible(512 * channel_ratio),
            kernel_size=2, stride=2, padding=0)
        self.right_layer_1 = DoubleConv(make_divisible(1024 * channel_ratio), make_divisible(512 * channel_ratio))
        self.up_sampling_2 = nn.ConvTranspose2d(
            make_divisible(512 * channel_ratio), make_divisible(256 * channel_ratio),
            kernel_size=2, stride=2, padding=0)
        self.right_layer_2 = DoubleConv(make_divisible(512 * channel_ratio), make_divisible(256 * channel_ratio))
        self.up_sampling_3 = nn.ConvTranspose2d(
            make_divisible(256 * channel_ratio), make_divisible(128 * channel_ratio),
            kernel_size=2, stride=2, padding=0)
        self.right_layer_3 = DoubleConv(make_divisible(256 * channel_ratio), make_divisible(128 * channel_ratio))
        self.up_sampling_4 = nn.ConvTranspose2d(
            make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio),
            kernel_size=2, stride=2, padding=0)
        self.right_layer_4 = DoubleConv(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio))

        self.final_layer = nn.Conv2d(make_divisible(64 * channel_ratio), classes,
                                     kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.left_layer_1(x)
        left_1 = x

        x = self.down_sampling(x)
        x = self.left_layer_2(x)
        left_2 = x

        x = self.down_sampling(x)
        x = self.left_layer_3(x)
        left_3 = x

        x = self.down_sampling(x)
        x = self.left_layer_4(x)
        left_4 = x

        x = self.down_sampling(x)
        x = self.middle_layer(x)

        x = self.up_sampling_1(x)
        x = torch.cat((x, left_4), dim=1)
        x = self.right_layer_1(x)

        x = self.up_sampling_2(x)
        x = torch.cat((x, left_3), dim=1)
        x = self.right_layer_2(x)

        x = self.up_sampling_3(x)
        x = torch.cat((x, left_2), dim=1)
        x = self.right_layer_3(x)

        x = self.up_sampling_4(x)
        x = torch.cat((x, left_1), dim=1)
        x = self.right_layer_4(x)

        x = self.final_layer(x)

        return x


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = UNet(in_channels=3, classes=10, channel_ratio=1.0)
    out_data = model(in_data)
