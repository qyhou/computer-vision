import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class ConvBlock(nn.Module):
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


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sampling_mode='transpose'):
        super().__init__()
        if up_sampling_mode == 'transpose':
            self.up_sampling = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            if in_channels == out_channels:
                self.up_sampling = nn.Upsample(scale_factor=2, mode=up_sampling_mode)
            else:
                self.up_sampling = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=up_sampling_mode),
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.up_sampling(x)
        return x


class UNet(nn.Module):
    def __init__(self, classes=21, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0):
        super().__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_layer_1 = ConvBlock(in_channels, make_divisible(64 * channel_ratio))
        self.left_layer_2 = ConvBlock(make_divisible(64 * channel_ratio), make_divisible(128 * channel_ratio))
        self.left_layer_3 = ConvBlock(make_divisible(128 * channel_ratio), make_divisible(256 * channel_ratio))
        self.left_layer_4 = ConvBlock(make_divisible(256 * channel_ratio), make_divisible(512 * channel_ratio))

        self.middle_layer = ConvBlock(make_divisible(512 * channel_ratio), make_divisible(1024 * channel_ratio))

        self.up_sampling_1 = UpSamplingBlock(make_divisible(1024 * channel_ratio), make_divisible(512 * channel_ratio),
                                             up_sampling_mode)
        self.right_layer_1 = ConvBlock(make_divisible(1024 * channel_ratio), make_divisible(512 * channel_ratio))
        self.up_sampling_2 = UpSamplingBlock(make_divisible(512 * channel_ratio), make_divisible(256 * channel_ratio),
                                             up_sampling_mode)
        self.right_layer_2 = ConvBlock(make_divisible(512 * channel_ratio), make_divisible(256 * channel_ratio))
        self.up_sampling_3 = UpSamplingBlock(make_divisible(256 * channel_ratio), make_divisible(128 * channel_ratio),
                                             up_sampling_mode)
        self.right_layer_3 = ConvBlock(make_divisible(256 * channel_ratio), make_divisible(128 * channel_ratio))
        self.up_sampling_4 = UpSamplingBlock(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio),
                                             up_sampling_mode)
        self.right_layer_4 = ConvBlock(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio))

        self.final_layer = nn.Conv2d(make_divisible(64 * channel_ratio), classes,
                                     kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        left_1 = self.left_layer_1(x)
        left_2 = self.left_layer_2(self.down_sampling(left_1))
        left_3 = self.left_layer_3(self.down_sampling(left_2))
        left_4 = self.left_layer_4(self.down_sampling(left_3))

        x = self.middle_layer(self.down_sampling(left_4))

        x = self.right_layer_1(torch.cat((self.up_sampling_1(x), left_4), dim=1))
        x = self.right_layer_2(torch.cat((self.up_sampling_2(x), left_3), dim=1))
        x = self.right_layer_3(torch.cat((self.up_sampling_3(x), left_2), dim=1))
        x = self.right_layer_4(torch.cat((self.up_sampling_4(x), left_1), dim=1))

        x = self.final_layer(x)

        return x


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = UNet(classes=21, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
    out_data = model(in_data)
    print(out_data.shape)
