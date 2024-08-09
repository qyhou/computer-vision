class Model:
    def __init__(self):
        self.model = None

    def calc_model_params(self):
        if self.model is None:
            print('model not defined')
            return
        total_params = sum(
            param.numel() for param in self.model.parameters()
        )
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        print('total_params: {:,}'.format(total_params))
        print('trainable_params: {:,}'.format(trainable_params))


class Backbone(Model):
    def __init__(self):
        super().__init__()

    def load_resnet(self):
        from models.backbones.resnet import resnet_18
        self.model = resnet_18(classes=1000, in_channels=3, channel_ratio=1.0)

    def load_vgg(self):
        from models.backbones.vgg import vgg16
        self.model = vgg16(classes=1000, in_channels=3, dropout=0.5, batch_norm=False, channel_ratio=1.0)


class Segmentation(Model):
    def __init__(self):
        super().__init__()

    def load_unet3p(self):
        from models.segmentation.unet3p import UNet3P
        self.model = UNet3P(classes=21, in_channels=3, channel_ratio=1.0)

    def load_unetpp(self):
        from models.segmentation.unetpp import UNetPP
        self.model = UNetPP(level=4, classes=21, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
        # self.model = UNetPP(level=4, classes=21, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)

    def load_unet(self):
        from models.segmentation.unet import UNet
        self.model = UNet(classes=21, in_channels=3, up_sampling_mode='transpose', channel_ratio=1.0)
        # self.model = UNet(classes=21, in_channels=3, up_sampling_mode='bilinear', channel_ratio=1.0)

    def load_fcn(self):
        from models.segmentation.fcn import FCN
        self.model = FCN(classes=21, in_channels=512, up_sampling_mode='transpose', up_sampling_factor=32)
        # self.model = FCN(classes=21, in_channels=512, up_sampling_mode='bilinear', up_sampling_factor=32)


if __name__ == '__main__':
    model = Backbone()
    model.load_resnet()
    model.calc_model_params()
