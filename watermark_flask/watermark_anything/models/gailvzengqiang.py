import torch
import torch.nn as nn
import torch.fft

class ExplicitLowFreqAttention(nn.Module):
    def __init__(self, height, width, alpha=1.0, learnable=False):
        super().__init__()
        self.height = height
        self.width = width
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=learnable)

        # 构建二维高斯分布的低频 mask（中心高，远离中心低）
        y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
        self.register_buffer('lowfreq_mask', torch.exp(-(x**2 + y**2) * 10))  # 可调“10”

    def forward(self, x):
        """
        x: [B, C, H, W] 图像空间
        return: 低频增强后图像
        """
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)

        # 增强低频部分（复数乘以低频 mask）
        mask = 1 + self.alpha * self.lowfreq_mask  # [H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        fft_real = fft_shift.real * mask
        fft_imag = fft_shift.imag * mask
        enhanced_fft = torch.complex(fft_real, fft_imag)

        # 逆变换回图像
        fft_ishift = torch.fft.ifftshift(enhanced_fft)
        out = torch.fft.ifft2(fft_ishift).real

        return out
