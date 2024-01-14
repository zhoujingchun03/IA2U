from torch import nn


class DownScalex2(nn.Module):
    def __init__(self, type, in_c=24):
        super().__init__()

        if type == 'AvgPool':
            proj = nn.AvgPool2d(kernel_size=2, stride=2)
        elif type == 'Conv':
            proj = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1, stride=2, bias=False)
        elif type == 'PixelUnshuffle':
            proj = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=in_c // 4, kernel_size=1, bias=False),
                nn.PixelUnshuffle(downscale_factor=2)
            )
        else:
            raise ValueError('ERROR TYPE: {}'.format(type))
        setattr(self, 'proj', proj)

    def forward(self, x):
        x = self.proj(x)
        return x


class UpScalex2(nn.Module):
    def __init__(self, type, in_c=24):
        super().__init__()

        if type == 'BiLinear':
            proj = nn.UpsamplingBilinear2d(scale_factor=2)
        elif type == 'TransConv':
            proj = nn.ConvTranspose2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=False)
        elif type == 'PixelShuffle':
            proj = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=in_c * 4, kernel_size=1, bias=False),
                nn.PixelShuffle(upscale_factor=2)
            )
        else:
            raise ValueError('ERROR TYPE: {}'.format(type))
        setattr(self, 'proj', proj)

    def forward(self, x):
        y = self.proj(x)
        return y


class CA(nn.Module):
    def __init__(self, channel=24,reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)