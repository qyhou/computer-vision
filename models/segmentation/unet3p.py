import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class Encoder(nn.Module):
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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, up=None):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.MaxPool2d(kernel_size=down, stride=down))
        elif up:
            layers.append(nn.Upsample(scale_factor=up, mode='bilinear'))
        layers.extend([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x


class UNet3P(nn.Module):
    def __init__(self, classes=10, in_channels=3, channel_ratio=1.0):
        super().__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_1 = Encoder(in_channels, make_divisible(64 * channel_ratio))
        self.encoder_2 = Encoder(make_divisible(64 * channel_ratio), make_divisible(128 * channel_ratio))
        self.encoder_3 = Encoder(make_divisible(128 * channel_ratio), make_divisible(256 * channel_ratio))
        self.encoder_4 = Encoder(make_divisible(256 * channel_ratio), make_divisible(512 * channel_ratio))
        self.encoder_5 = Encoder(make_divisible(512 * channel_ratio), make_divisible(1024 * channel_ratio))

        self.decoder_4_1 = Decoder(make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), down=8)
        self.decoder_4_2 = Decoder(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio), down=4)
        self.decoder_4_3 = Decoder(make_divisible(256 * channel_ratio), make_divisible(64 * channel_ratio), down=2)
        self.decoder_4_4 = Decoder(make_divisible(512 * channel_ratio), make_divisible(64 * channel_ratio))
        self.decoder_4_5 = Decoder(make_divisible(1024 * channel_ratio), make_divisible(64 * channel_ratio), up=2)
        self.decoder_4_fusion = Decoder(make_divisible(320 * channel_ratio), make_divisible(320 * channel_ratio))

        self.decoder_3_1 = Decoder(make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), down=4)
        self.decoder_3_2 = Decoder(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio), down=2)
        self.decoder_3_3 = Decoder(make_divisible(256 * channel_ratio), make_divisible(64 * channel_ratio))
        self.decoder_3_4 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=2)
        self.decoder_3_5 = Decoder(make_divisible(1024 * channel_ratio), make_divisible(64 * channel_ratio), up=4)
        self.decoder_3_fusion = Decoder(make_divisible(320 * channel_ratio), make_divisible(320 * channel_ratio))

        self.decoder_2_1 = Decoder(make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio), down=2)
        self.decoder_2_2 = Decoder(make_divisible(128 * channel_ratio), make_divisible(64 * channel_ratio))
        self.decoder_2_3 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=2)
        self.decoder_2_4 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=4)
        self.decoder_2_5 = Decoder(make_divisible(1024 * channel_ratio), make_divisible(64 * channel_ratio), up=8)
        self.decoder_2_fusion = Decoder(make_divisible(320 * channel_ratio), make_divisible(320 * channel_ratio))

        self.decoder_1_1 = Decoder(make_divisible(64 * channel_ratio), make_divisible(64 * channel_ratio))
        self.decoder_1_2 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=2)
        self.decoder_1_3 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=4)
        self.decoder_1_4 = Decoder(make_divisible(320 * channel_ratio), make_divisible(64 * channel_ratio), up=8)
        self.decoder_1_5 = Decoder(make_divisible(1024 * channel_ratio), make_divisible(64 * channel_ratio), up=16)
        self.decoder_1_fusion = Decoder(make_divisible(320 * channel_ratio), make_divisible(320 * channel_ratio))

        self.final_layer_5 = nn.Sequential(
            nn.Conv2d(make_divisible(1024 * channel_ratio), classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=16, mode='bilinear')
        )
        self.final_layer_4 = nn.Sequential(
            nn.Conv2d(make_divisible(320 * channel_ratio), classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=8, mode='bilinear')
        )
        self.final_layer_3 = nn.Sequential(
            nn.Conv2d(make_divisible(320 * channel_ratio), classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.final_layer_2 = nn.Sequential(
            nn.Conv2d(make_divisible(320 * channel_ratio), classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.final_layer_1 = nn.Sequential(
            nn.Conv2d(make_divisible(320 * channel_ratio), classes,
                      kernel_size=3, stride=1, padding=1)
        )

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(make_divisible(1024 * channel_ratio), 2,
                      kernel_size=1, stride=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )

    @staticmethod
    def _dot_product(seg, cls):
        b, c, h, w = seg.size()
        seg = seg.view(b, c, h * w)
        x = torch.einsum("ijk,ij->ijk", [seg, cls])
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        e_1 = self.encoder_1(x)
        e_2 = self.encoder_2(self.down_sampling(e_1))
        e_3 = self.encoder_3(self.down_sampling(e_2))
        e_4 = self.encoder_4(self.down_sampling(e_3))
        e_5 = self.encoder_5(self.down_sampling(e_4))
        d_5 = e_5

        d_4_1 = self.decoder_4_1(e_1)
        d_4_2 = self.decoder_4_2(e_2)
        d_4_3 = self.decoder_4_3(e_3)
        d_4_4 = self.decoder_4_4(e_4)
        d_4_5 = self.decoder_4_5(d_5)
        d_4 = self.decoder_4_fusion(torch.cat((d_4_1, d_4_2, d_4_3, d_4_4, d_4_5), dim=1))

        d_3_1 = self.decoder_3_1(e_1)
        d_3_2 = self.decoder_3_2(e_2)
        d_3_3 = self.decoder_3_3(e_3)
        d_3_4 = self.decoder_3_4(d_4)
        d_3_5 = self.decoder_3_5(d_5)
        d_3 = self.decoder_3_fusion(torch.cat((d_3_1, d_3_2, d_3_3, d_3_4, d_3_5), dim=1))

        d_2_1 = self.decoder_2_1(e_1)
        d_2_2 = self.decoder_2_2(e_2)
        d_2_3 = self.decoder_2_3(d_3)
        d_2_4 = self.decoder_2_4(d_4)
        d_2_5 = self.decoder_2_5(d_5)
        d_2 = self.decoder_2_fusion(torch.cat((d_2_1, d_2_2, d_2_3, d_2_4, d_2_5), dim=1))

        d_1_1 = self.decoder_1_1(e_1)
        d_1_2 = self.decoder_1_2(d_2)
        d_1_3 = self.decoder_1_3(d_3)
        d_1_4 = self.decoder_1_4(d_4)
        d_1_5 = self.decoder_1_5(d_5)
        d_1 = self.decoder_1_fusion(torch.cat((d_1_1, d_1_2, d_1_3, d_1_4, d_1_5), dim=1))

        d_5_deep_supervision = self.final_layer_5(d_5)
        d_4_deep_supervision = self.final_layer_4(d_4)
        d_3_deep_supervision = self.final_layer_3(d_3)
        d_2_deep_supervision = self.final_layer_2(d_2)
        d_1_deep_supervision = self.final_layer_1(d_1)

        cls_branch = self.cls(d_5).squeeze(3).squeeze(2)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max.unsqueeze(1).float()

        x_1 = self._dot_product(d_1_deep_supervision, cls_branch_max)
        x_2 = self._dot_product(d_2_deep_supervision, cls_branch_max)
        x_3 = self._dot_product(d_3_deep_supervision, cls_branch_max)
        x_4 = self._dot_product(d_4_deep_supervision, cls_branch_max)
        x_5 = self._dot_product(d_5_deep_supervision, cls_branch_max)

        return x_5, x_4, x_3, x_2, x_1


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 960, 640)  # b, c, h, w
    model = UNet3P(classes=10, in_channels=3, channel_ratio=1.0)
    out_data = model(in_data)
