from torch import nn
import torch
from torchinfo import summary

IMG_SIZE = (384, 384)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x, concat=None):
        if concat is None:
            return self.block(x)
        else:
            return self.block(torch.concat([x, concat], dim=1))


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # downsize
        # 384
        self.block1 = Block(1, 32, 5, 2)
        # 192
        self.block2 = Block(32, 64, 3, 1)
        # 96
        self.block3 = Block(64, 64, 3, 1)
        # 48
        self.block4 = Block(64, 128, 3, 1)
        # 24
        self.bottleneck = Block(128, 128, 3, 1)

        # General poolers
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # upsize
        # 48
        self.upsample1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.rblock1 = Block(128+128, 128, 3, 1)
        # 96
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.rblock2 = Block(64+128, 64, 3, 1)
        # 192
        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.rblock3 = Block(64+64, 64, 3, 1)
        # 384
        self.upsample4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.rblock4 = Block(64+32, 1, 5, 2)

    def forward(self, x):
        # downsampling
        y1 = self.block1(x)
        x1 = self.maxpool(y1)

        y2 = self.block2(x1)
        x2 = self.maxpool(y2)

        y3 = self.block3(x2)
        x3 = self.maxpool(y3)

        y4 = self.block4(x3)
        x4 = self.maxpool(y4)

        x4 = self.bottleneck(x4)

        # upsampling
        y5 = self.upsample1(x4)
        # print(y5.shape, y4.shape)
        x5 = self.rblock1(y5, concat=y4)

        y6 = self.upsample2(x5)
        x6 = self.rblock2(y6, concat=y3)

        y7 = self.upsample3(x6)
        x7 = self.rblock3(y7, concat=y2)

        y8 = self.upsample4(x7)
        x8 = self.rblock4(y8, concat=y1)

        return x8


class UNETUpsampleBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        # downsize
        # 384
        self.block1 = Block(1, 32, 5, 2)
        # 192
        self.block2 = Block(32, 64, 3, 1)
        # 96
        self.block3 = Block(64, 64, 3, 1)
        # 48
        self.block4 = Block(64, 128, 3, 1)
        # 24
        self.bottleneck = Block(128, 128, 3, 1)

        # General poolers
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # upsize
        # 48
        self.rblock1 = Block(128+128, 128, 3, 1)
        # 96
        self.rblock2 = Block(64+128, 64, 3, 1)
        # 192
        self.rblock3 = Block(64+64, 64, 3, 1)
        # 384
        self.rblock4 = Block(64+32, 1, 5, 2)

    def forward(self, x):
        # downsampling
        y1 = self.block1(x)
        x1 = self.maxpool(y1)

        y2 = self.block2(x1)
        x2 = self.maxpool(y2)

        y3 = self.block3(x2)
        x3 = self.maxpool(y3)

        y4 = self.block4(x3)
        x4 = self.maxpool(y4)

        x4 = self.bottleneck(x4)

        # upsampling
        y5 = self.upsample(x4)
        # print(y5.shape, y4.shape)
        x5 = self.rblock1(y5, concat=y4)

        y6 = self.upsample(x5)
        x6 = self.rblock2(y6, concat=y3)

        y7 = self.upsample(x6)
        x7 = self.rblock3(y7, concat=y2)

        y8 = self.upsample(x7)
        x8 = self.rblock4(y8, concat=y1)

        return x8

if __name__ == "__main__":
    model = UNETUpsampleBilinear()
    print(summary(model, input_size=(3, 1, 384, 384)))