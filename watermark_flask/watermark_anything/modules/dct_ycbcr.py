import math
import torch
import numpy as np
from einops import rearrange
from PIL import Image, ImageOps

class ColorDCTTransform:
    def __init__(self, device=None):
        """
        :param device: 设备，如 torch.device('cuda:0') 或 torch.device('cpu')
        """
        # 如果调用时未指定，则默认为 CPU
        self.device = device if device is not None else torch.device("cpu")

        # 预先构造 8x8 DCT 矩阵及其转置和逆，用于 dct_dxc / idct_dxc
        self.dctm = self.getDCTM((8, 8)).to(self.device)
        self.dctm_T = self.dctm.transpose(0, 1)
        self.dctm_inv = torch.inverse(self.dctm)
        self.dctm_T_inv = torch.inverse(self.dctm_T)

    def ycbcr2rgb(self, input_im):
        """
        将 YCbCr 转换为 RGB
        :param input_im: (3,H,W) tensor
        :return: (3,H,W) RGB tensor
        """
        # 确保数据在 self.device 上
        input_im = input_im.to(self.device)

        im_flat = input_im.contiguous().view(-1, 3).float()

        bias = torch.tensor([16.0, 128.0, 128.0], device=self.device)
        y_range = torch.tensor([219.0, 224.0, 224.0], device=self.device)
        mat = torch.tensor([
            [1.0, -1.21889419e-06, 1.40199959],
            [1.0, -3.44135678e-01, -7.14136156e-01],
            [1.0, 1.77200007e00, 4.06298063e-07]
        ], device=self.device)

        im_flat = im_flat - bias
        im_flat = im_flat / y_range
        temp = im_flat.mm(mat.t())
        out = temp.view(3, input_im.size(1), input_im.size(2))
        return out

    def rgb2ycbcr(self, input_im):
        """
        将 RGB 转换为 YCbCr
        :param input_im: (3,H,W) tensor
        :return: (3,H,W)
        """
        input_im = input_im.to(self.device)
        im_flat = input_im.contiguous().view(-1, 3).float()

        mat = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ], device=self.device)
        bias = torch.tensor([16.0, 128.0, 128.0], device=self.device)
        y_range = torch.tensor([219.0, 224.0, 224.0], device=self.device)

        temp = im_flat.mm(mat.t())
        temp = temp * y_range
        temp = temp + bias
        out = temp.view(3, input_im.shape[1], input_im.shape[2])
        return out

    def my_rgb2ycbcr(self, im):
        """
        自定义的 RGB->YCbCr
        :param im: (3,H,W) tensor
        :return: (3,H,W)
        """
        im = im.to(self.device)

        r = im[0, :, :]
        g = im[1, :, :]
        b = im[2, :, :]

        Y  = (r *  65.481 + g * 128.553 + b *  24.966 +  16.0).unsqueeze(0)
        Cb = (r * -37.797 - g *  74.203 + b * 112.0    + 128.0).unsqueeze(0)
        Cr = (r * 112.0   - g *  93.786 - b *  18.214  + 128.0).unsqueeze(0)

        return torch.cat([Y, Cb, Cr], dim=0)

    def my_ycbcr2rgb(self, ycbcr):
        """
        自定义的 YCbCr->RGB
        :param ycbcr: (3,H,W)
        :return: (3,H,W)
        """
        ycbcr = ycbcr.to(self.device)

        Y  = (ycbcr[0, :, :] - 16.0)   / 219.0
        Cb = (ycbcr[1, :, :] - 128.0) / 224.0
        Cr = (ycbcr[2, :, :] - 128.0) / 224.0

        R = (Y + Cb * -1.21889419e-06 + Cr *  1.40199959).unsqueeze(0)
        G = (Y + Cb * -3.44135678e-01 + Cr * -7.14136156e-01).unsqueeze(0)
        B = (Y + Cb *  1.77200007     + Cr *  4.06298063e-07).unsqueeze(0)

        return torch.cat([R, G, B], dim=0)

    def rgb2ycbcr_batch(self, images):
        """
        batch 版的 RGB->YCbCr
        :param images: (N,3,H,W)
        """
        images = images.to(self.device)

        r = images[:, 0, :, :]
        g = images[:, 1, :, :]
        b = images[:, 2, :, :]

        Y  = (r *  65.481 + g * 128.553 + b *  24.966 + 16.0 ).unsqueeze(1)
        Cb = (r * -37.797 - g *  74.203 + b * 112.0    + 128.0).unsqueeze(1)
        Cr = (r * 112.0   - g *  93.786 - b *  18.214  + 128.0).unsqueeze(1)

        return torch.cat([Y, Cb, Cr], dim=1)

    def ycbcr2rgb_batch(self, images):
        """
        batch 版的 YCbCr->RGB
        :param images: (N,3,H,W)
        """
        images = images.to(self.device)

        Y  = (images[:, 0, :, :] - 16.0)   / 219.0
        Cb = (images[:, 1, :, :] - 128.0) / 224.0
        Cr = (images[:, 2, :, :] - 128.0) / 224.0

        R = (Y + Cb * -1.21889419e-06 + Cr *  1.40199959).unsqueeze(1)
        G = (Y + Cb * -3.44135678e-01 + Cr * -7.14136156e-01).unsqueeze(1)
        B = (Y + Cb *  1.77200007     + Cr *  4.06298063e-07).unsqueeze(1)

        return torch.cat([R, G, B], dim=1)

    def getDCTM(self, size):
        """
        构造 DCT 变换矩阵
        :param size: (height, width)
        :return: (height, width) dct 矩阵
        """
        h, w = size
        dctm = torch.ones((h, w), device=self.device)
        for i in range(h):
            for j in range(w):
                c = math.sqrt(1.0 / h) if i == 0 else math.sqrt(2.0 / h)
                dctm[i, j] = c * math.cos((j + 0.5) * math.pi * i / h)
        return dctm

    def my_dct2d(self, input_tensor):
        """
        对 (b,8,8) 做 DCT
        :param input_tensor: (b,8,8)
        """
        input_tensor = input_tensor.to(self.device)
        # 使用批量矩阵乘法
        temp = torch.bmm(
            self.dctm.unsqueeze(0).expand(input_tensor.size(0), -1, -1),
            input_tensor
        )
        ans = torch.bmm(
            temp,
            self.dctm_T.unsqueeze(0).expand(input_tensor.size(0), -1, -1)
        )
        return ans

    def my_idct2d(self, input_tensor):
        """
        对 (b,8,8) 做逆 DCT
        """
        input_tensor = input_tensor.to(self.device)
        temp = torch.bmm(
            self.dctm_inv.unsqueeze(0).expand(input_tensor.size(0), -1, -1),
            input_tensor
        )
        ans = torch.bmm(
            temp,
            self.dctm_T_inv.unsqueeze(0).expand(input_tensor.size(0), -1, -1)
        )
        return ans

    def dct_dxc(self, imgs):
        """
        对 (b, H, W) 做分块 DCT，每块 8x8
        :param imgs: (b,H,W)
        :return: (b, 64, H/8, W/8)
        """
        imgs = imgs.to(self.device)
        # 重排成小块 (b*h*w, 8, 8)
        item_list = rearrange(imgs, 'b (h p1) (w p2) -> (b h w) p1 p2', p1=8, p2=8)
        # 对每个 8x8 小块做 DCT
        item_list = self.my_dct2d(item_list)
        # 还原回 (b, 64, H/8, W/8)
        item_list = rearrange(
            item_list,
            '(b h w) p1 p2 -> b (p1 p2) h w',
            h=imgs.shape[1] // 8, w=imgs.shape[2] // 8
        )
        return item_list

    def idct_dxc(self, imgs):
        """
        对 (b, 64, H/8, W/8) 做分块 iDCT
        :return: (b, H, W)
        """
        imgs = imgs.to(self.device)
        item_list = rearrange(imgs, 'b (p1 p2) h w -> (b h w) p1 p2', p1=8, p2=8)
        item_list = self.my_idct2d(item_list)
        item_list = rearrange(
            item_list,
            '(b h w) p1 p2 -> b (h p1) (w p2)',
            h=imgs.shape[2], w=imgs.shape[3]
        )
        return item_list

    def dwt(self, x):
        """
        简易 2D 离散小波变换
        x: (b, c, h, w)
        返回 (b, c*4, h//2, w//2)
        """
        x = x.to(self.device)
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def iwt(self, x):
        """
        简易 2D 离散小波逆变换
        :param x: (b, c*4, h, w)
        :return: (b, c, 2h, 2w)
        """
        x = x.to(self.device)
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = in_channel // (r ** 2)
        out_height = r * in_height
        out_width = r * in_width

        h = torch.zeros(
            [in_batch, out_channel, out_height, out_width],
            dtype=x.dtype,
            device=self.device
        )

        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


if __name__ == '__main__':
    # 使用示例：

    # 1) 指定我们要在 GPU 或 CPU 上执行
    # device = torch.device('cuda:0')  # 如果有 GPU
    device = torch.device('cpu')       # 如果只在 CPU 上测试

    # 2) 创建类的实例
    transform = ColorDCTTransform(device=device)

    # 3) 随机创建一张图片 (3, H, W)，例如 (3,400,400)
    # 这里用纯灰色图来演示
    img_cover = Image.new("RGB", (400, 400), (128, 128, 128))
    img_cover = ImageOps.fit(img_cover, (400, 400))
    img_cover = np.array(img_cover, dtype=np.float32) / 255.0
    img_cover = np.transpose(img_cover, (2, 0, 1))  # (3,H,W)
    img_cover = torch.from_numpy(img_cover).to(device)

    # 4) 测试自定义 YCbCr->RGB
    ycbcr = transform.my_rgb2ycbcr(img_cover)
    rgb_back = transform.my_ycbcr2rgb(ycbcr)

    # 5) 转回 numpy 看一下
    rgb_np = rgb_back.cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0)) * 255.
    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

    # 6) 保存结果
    out_img = Image.fromarray(rgb_np)
    out_img.save("out_test.jpg")

    print("Done.")
