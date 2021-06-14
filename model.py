import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GenUnet(nn.Module):
    def __init__(self, bilinear=True):
        super(GenUnet, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Dis(torch.nn.Module):
    def __init__(self):
        super(Dis, self).__init__()

        self.output_shape = (1, 64, 64)

        self.dis = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            #M-1
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-2
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-3
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-4
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-5
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-6
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-7
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-8
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

class VGG19_fea(torch.nn.Module):
    def __init__(self):
        super(VGG19_fea, self).__init__()
        self.conv1to3 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        model = models.vgg19_bn(pretrained=True)
        self.vgg19_fea = model.features

    def forward(self, x):
        x = self.conv1to3(x)
        x = self.relu(x)
        x = self.vgg19_fea(x)

        return(x)

if __name__ == "__main__":
   device = "cuda" if torch.cuda.is_available() else "cpu"
   x = torch.randn(1 ,1, 256, 256).to(device)
   model = GenUnet().to(device)
   print(model(x).shape)

