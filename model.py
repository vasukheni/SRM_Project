import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x) * 0.1


class SRGenerator(nn.Module):

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        num_res_blocks=6,
        scale=4
    ):
        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(
            in_channels,
            base_channels,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(base_channels)
                for _ in range(num_res_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            base_channels,
            base_channels,
            3,
            1,
            1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels * 4,
                3,
                1,
                1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels,
                base_channels * 4,
                3,
                1,
                1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            base_channels,
            in_channels,
            3,
            1,
            1
        )

    def forward(self, x):

        bicubic = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        feat = self.head(x)

        body = self.body(feat)
        body = self.body_conv(body)

        feat = feat + body

        feat = self.upsample(feat)

        residual = self.tail(feat)

        output = bicubic + residual

        return output.clamp(0, 1)