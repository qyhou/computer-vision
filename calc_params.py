from models.backbones.resnet import resnet_18
from models.segmentation.unet import UNet
# from models.segmentation.unetpp import UNetPP
from models.segmentation.unet3p import UNet3P

# model = resnet_18(in_channels=3, channel_ratio=1.0)
# model = UNet(classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
# model = UNet(classes=10, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)
# model = UNetPP(level=4, classes=10, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
# model = UNetPP(level=4, classes=10, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)
model = UNet3P(classes=10, in_channels=3, channel_ratio=1.0)

total_params = sum(
    param.numel() for param in model.parameters()
)

trainable_params = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)

print('total_params: {:,}'.format(total_params))
print('trainable_params: {:,}'.format(trainable_params))
