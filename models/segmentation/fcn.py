import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, classes=21, in_channels=512, up_sampling_mode='transpose', up_sampling_factor=32):
        super().__init__()
        self.final_conv = nn.Conv2d(in_channels, classes, kernel_size=1, stride=1, padding=0)
        if up_sampling_mode == 'transpose':
            self.up_sampling = nn.ConvTranspose2d(
                classes, classes,
                kernel_size=up_sampling_factor * 2, stride=up_sampling_factor, padding=up_sampling_factor // 2)
        else:
            self.up_sampling = nn.Upsample(scale_factor=up_sampling_factor, mode=up_sampling_mode)

    def forward(self, x):
        x = self.final_conv(x)
        x = self.up_sampling(x)
        return x


if __name__ == '__main__':
    in_data = torch.randn(1, 512, 30, 20)  # b, c, h, w
    model = FCN(classes=21, in_channels=512, up_sampling_mode='transpose', up_sampling_factor=32)
    out_data = model(in_data)
    print(out_data.shape)
