import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    基础的卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    不改变输入输出的通道数
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """
    上采样模块：Transposed Conv(2x上采样) + DoubleConv
    注意：输入输出通道均为 in_ch，不改变通道数
    """
    def __init__(self, in_ch):
        super().__init__()
        # 2倍上采样 (转置卷积)
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch)

    def forward(self, x):
        # B,in_ch,H/2,W/2 -> B,in_ch,H,W
        x = self.up(x)
        x = self.conv(x)
        return x


class UNetFreqNoChannelChange(nn.Module):
    """
    在频域输出 (B,64,H/8,W/8) 的特征图上采样 8 倍到 (B,out_channels,H,W)，
    中间通道数保持为 64 不变。
    适合在 "频域" 做逐像素水印检测/解码时使用。
    """
    def __init__(self):
        """
        out_channels: 最终输出通道数，比如 1+nbits
        """
        super().__init__()
        # 首先用 DoubleConv 初步处理一下 (64->64)
        self.inc = DoubleConv(64)

        # 需要 3 次上采样，每次 2x，因此 2*2*2=8
        self.up1 = UpBlock(64)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(64)



    def forward(self, x):
        """
        x: 形状 (B,64,H/8,W/8)，H/8 和 W/8 需满足可被上采样回 H×W
        Returns: (B,out_channels,H,W)
        """
        # 先用 DoubleConv 深加工一下特征
        x = self.inc(x)

        # 连续三次上采样: (H/8)->(H/4)->(H/2)->(H)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return x




