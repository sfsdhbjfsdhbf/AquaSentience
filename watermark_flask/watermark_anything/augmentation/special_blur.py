

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelateAugmentation(nn.Module):
    def __init__(self, block_size=(8,8)):
        super(PixelateAugmentation,self).__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor,mask) -> torch.Tensor:
        b, c, h, w = x.shape
        block_h, block_w = self.block_size

        h_out = h // block_h
        w_out = w // block_w

        x = x.view(b, c, h_out, block_h, w_out, block_w)
        x = x.mean(dim=(3, 5))
        x = F.interpolate(x, size=(h, w), mode='nearest')
        return x,mask


class DefocusBlurAugmentation(nn.Module):
    def __init__(self, sigma: tuple = (2.0, 5.0)):
        super(DefocusBlurAugmentation,self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor,mask) -> torch.Tensor:
        kernel_size = int(2 * torch.ceil(3 * torch.tensor(self.sigma[1])).item()) + 1
        kernel = self.create_disk_kernel(kernel_size, self.sigma[1])
        kernel = (kernel / kernel.sum()).to(x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # [C, 1, kH, kW]
        return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1]),mask

    def create_disk_kernel(self, size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(-size // 2 + 1, size // 2 + 1)
        y = torch.arange(-size // 2 + 1, size // 2 + 1)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # 避免警告
        kernel = (xx ** 2 + yy ** 2) <= (size // 2) ** 2
        return kernel.float()




import torch
import torchvision.transforms.functional as tvf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# 定义测试图像路径
TEST_IMAGE_PATH = "/media/wh/MyDist/ywy/SSyncOA-main/datasets/DUTS/DUTS-TE/Std-Image-30/256/ILSVRC2012_test_00000026.jpg" # 替换为你的测试图像路径

# 加载测试图像并转换为张量
def load_test_image():
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_tsr = tvf.to_tensor(image).unsqueeze(0) # [1, 3, H, W]
    return image_tsr

# 测试 PixelateAugmentation
def test_pixelate_augmentation():
    image_tsr = load_test_image()
    print("=== Pixelate Augmentation Test ===")

    # 定义不同 block_size 参数
    block_sizes = [(8, 8), (16, 16)]

    for block_size in block_sizes:
        print(f"\nBlock Size: {block_size}")
        pixelate_aug = PixelateAugmentation(block_size=block_size)
        augmented = pixelate_aug(image_tsr)

    # 检查图像尺寸
    print(f"Input shape: {image_tsr.shape}, Output shape: {augmented.shape}")

    # 可视化
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image_tsr.squeeze().permute(1, 2, 0))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented.squeeze().permute(1, 2, 0))
    plt.title(f"Pixelate (block={block_size})")
    plt.axis('off')
    plt.show()

# 测试 DefocusBlurAugmentation
def test_defocus_blur_augmentation():
    image_tsr = load_test_image()
    print("\n=== Defocus Blur Augmentation Test ===")

    # 定义不同 sigma 参数
    sigmas = [(2.0, 5.0)]

    for sigma in sigmas:
        print(f"\nSigma: {sigma}")
        defocus_aug = DefocusBlurAugmentation(sigma=sigma)
        augmented = defocus_aug(image_tsr)

    # 检查图像尺寸
    print(f"Input shape: {image_tsr.shape}, Output shape: {augmented.shape}")

    # 可视化
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(augmented.squeeze().permute(1, 2, 0))
    plt.title("Original")
    plt.axis('off')
    plt.show()