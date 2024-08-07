from models.backbones.resnet import resnet_18
from models.backbones.vgg import vgg16

from models.segmentation.unet3p import UNet3P
from models.segmentation.unetpp import UNetPP
from models.segmentation.unet import UNet

# model = resnet_18(classes=1000, in_channels=3, channel_ratio=1.0)
model = vgg16(classes=1000, in_channels=3, dropout=0.5, channel_ratio=1.0)

# model = UNet3P(classes=10, in_channels=3, channel_ratio=1.0)
# model = UNetPP(level=4, classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
# model = UNetPP(level=4, classes=10, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)
# model = UNet(classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
# model = UNet(classes=10, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)

total_params = sum(
    param.numel() for param in model.parameters()
)

trainable_params = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)

print('total_params: {:,}'.format(total_params))
print('trainable_params: {:,}'.format(trainable_params))
