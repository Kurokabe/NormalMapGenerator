import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class UNet(nn.Module):
    def __init__(self, d=64, out_channels=3):
        super().__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 3, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 3, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 3, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # add an Upsample layer

        self.pad = nn.ReflectionPad2d(1)

        self.deconv1 = nn.Conv2d(d * 8, d * 8, 3, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.Conv2d(d * 8 * 2, d * 8, 3, 1, 0)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.Conv2d(d * 8 * 2, d * 8, 3, 1, 0)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.Conv2d(d * 8 * 2, d * 8, 3, 1, 0)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.Conv2d(d * 8 * 2, d * 4, 3, 1, 0)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.Conv2d(d * 4 * 2, d * 2, 3, 1, 0)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.Conv2d(d * 2 * 2, d, 3, 1, 0)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.last_conv = nn.Conv2d(d * 2, out_channels, 3, 1, 0)
        self.deconv8 = nn.Sequential(self.last_conv, nn.Tanh())

        self.weight_init(mean=0.0, std=0.02)

    def get_last_layer(self):
        return self.last_conv

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(
            self.deconv1_bn(self.deconv1(self.pad(self.up(F.relu(e8))))),
            0.5,
            training=True,
        )
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(
            self.deconv2_bn(self.deconv2(self.pad(self.up(F.relu(d1))))),
            0.5,
            training=True,
        )
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(
            self.deconv3_bn(self.deconv3(self.pad(self.up(F.relu(d2))))),
            0.5,
            training=True,
        )
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(self.pad(self.up(F.relu(d3)))))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(self.pad(self.up(F.relu(d4)))))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(self.pad(self.up(F.relu(d5)))))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(self.pad(self.up(F.relu(d6)))))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(self.pad(self.up(F.relu(d7))))
        # o = F.tanh(d8)

        return d8


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
