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


class UNetPP(nn.Module):
    def __init__(self, level=4, classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0):
        super().__init__()
        self.level = level

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_0_0 = ConvBlock(in_channels, make_divisible(32 * channel_ratio))
        self.layer_1_0 = ConvBlock(make_divisible(32 * channel_ratio), make_divisible(64 * channel_ratio))
        self.layer_0_1 = ConvBlock(make_divisible(96 * channel_ratio), make_divisible(32 * channel_ratio))
        self.up_sampling_1_0 = UpSamplingBlock(
            make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), up_sampling_mode)
        self.final_layer_1 = nn.Conv2d(make_divisible(32 * channel_ratio), classes,
                                       kernel_size=1, stride=1, padding=0)
        if level >= 2:
            self.layer_2_0 = ConvBlock(make_divisible(64 * channel_ratio), make_divisible(128 * channel_ratio))
            self.layer_1_1 = ConvBlock(make_divisible(192 * channel_ratio), make_divisible(64 * channel_ratio))
            self.layer_0_2 = ConvBlock(make_divisible(128 * channel_ratio), make_divisible(32 * channel_ratio))
            self.up_sampling_2_0 = UpSamplingBlock(
                make_divisible(128 * channel_ratio), make_divisible(128 * channel_ratio), up_sampling_mode)
            self.up_sampling_1_1 = UpSamplingBlock(
                make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), up_sampling_mode)
            self.final_layer_2 = nn.Conv2d(make_divisible(32 * channel_ratio), classes,
                                           kernel_size=1, stride=1, padding=0)
        if level >= 3:
            self.layer_3_0 = ConvBlock(make_divisible(128 * channel_ratio), make_divisible(256 * channel_ratio))
            self.layer_2_1 = ConvBlock(make_divisible(384 * channel_ratio), make_divisible(128 * channel_ratio))
            self.layer_1_2 = ConvBlock(make_divisible(256 * channel_ratio), make_divisible(64 * channel_ratio))
            self.layer_0_3 = ConvBlock(make_divisible(160 * channel_ratio), make_divisible(32 * channel_ratio))
            self.up_sampling_3_0 = UpSamplingBlock(
                make_divisible(256 * channel_ratio), make_divisible(256 * channel_ratio), up_sampling_mode)
            self.up_sampling_2_1 = UpSamplingBlock(
                make_divisible(128 * channel_ratio), make_divisible(128 * channel_ratio), up_sampling_mode)
            self.up_sampling_1_2 = UpSamplingBlock(
                make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), up_sampling_mode)
            self.final_layer_3 = nn.Conv2d(make_divisible(32 * channel_ratio), classes,
                                           kernel_size=1, stride=1, padding=0)
        if level >= 4:
            self.layer_4_0 = ConvBlock(make_divisible(256 * channel_ratio), make_divisible(512 * channel_ratio))
            self.layer_3_1 = ConvBlock(make_divisible(768 * channel_ratio), make_divisible(256 * channel_ratio))
            self.layer_2_2 = ConvBlock(make_divisible(512 * channel_ratio), make_divisible(128 * channel_ratio))
            self.layer_1_3 = ConvBlock(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio))
            self.layer_0_4 = ConvBlock(make_divisible(192 * channel_ratio), make_divisible(32 * channel_ratio))
            self.up_sampling_4_0 = UpSamplingBlock(
                make_divisible(512 * channel_ratio), make_divisible(512 * channel_ratio), up_sampling_mode)
            self.up_sampling_3_1 = UpSamplingBlock(
                make_divisible(256 * channel_ratio), make_divisible(256 * channel_ratio), up_sampling_mode)
            self.up_sampling_2_2 = UpSamplingBlock(
                make_divisible(128 * channel_ratio), make_divisible(128 * channel_ratio), up_sampling_mode)
            self.up_sampling_1_3 = UpSamplingBlock(
                make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), up_sampling_mode)
            self.final_layer_4 = nn.Conv2d(make_divisible(32 * channel_ratio), classes,
                                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_0_0 = self.layer_0_0(x)
        x_1_0 = self.layer_1_0(self.down_sampling(x_0_0))
        x_0_1 = self.layer_0_1(torch.cat((x_0_0, self.up_sampling_1_0(x_1_0)), dim=1))
        x_1 = self.final_layer_1(x_0_1)
        x_deep_supervision = x_1
        if self.level < 2:
            return x_deep_supervision

        x_2_0 = self.layer_2_0(self.down_sampling(x_1_0))
        x_1_1 = self.layer_1_1(torch.cat((x_1_0, self.up_sampling_2_0(x_2_0)), dim=1))
        x_0_2 = self.layer_0_2(torch.cat((x_0_0, x_0_1, self.up_sampling_1_1(x_1_1)), dim=1))
        x_2 = self.final_layer_2(x_0_2)
        x_deep_supervision += x_2
        if self.level < 3:
            return x_deep_supervision / self.level

        x_3_0 = self.layer_3_0(self.down_sampling(x_2_0))
        x_2_1 = self.layer_2_1(torch.cat((x_2_0, self.up_sampling_3_0(x_3_0)), dim=1))
        x_1_2 = self.layer_1_2(torch.cat((x_1_0, x_1_1, self.up_sampling_2_1(x_2_1)), dim=1))
        x_0_3 = self.layer_0_3(torch.cat((x_0_0, x_0_1, x_0_2, self.up_sampling_1_2(x_1_2)), dim=1))
        x_3 = self.final_layer_3(x_0_3)
        x_deep_supervision += x_3
        if self.level < 4:
            return x_deep_supervision / self.level

        x_4_0 = self.layer_4_0(self.down_sampling(x_3_0))
        x_3_1 = self.layer_3_1(torch.cat((x_3_0, self.up_sampling_4_0(x_4_0)), dim=1))
        x_2_2 = self.layer_2_2(torch.cat((x_2_0, x_2_1, self.up_sampling_3_1(x_3_1)), dim=1))
        x_1_3 = self.layer_1_3(torch.cat((x_1_0, x_1_1, x_1_2, self.up_sampling_2_2(x_2_2)), dim=1))
        x_0_4 = self.layer_0_4(torch.cat((x_0_0, x_0_1, x_0_2, x_0_3, self.up_sampling_1_3(x_1_3)), dim=1))
        x_4 = self.final_layer_4(x_0_4)
        x_deep_supervision += x_4
        return x_deep_supervision / self.level


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = UNetPP(level=4, classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
    out_data = model(in_data)
