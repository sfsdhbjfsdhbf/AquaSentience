# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import os
import sys
sys.path.append(os.path.abspath('/home/wh/ywy/watermark-anything-main/watermark_anything'))
import omegaconf

from modules.msg_processor import MsgProcessor
from modules.Ltransformer import CVTDecoder
from modules.dct_ycbcr import ColorDCTTransform
from modules.pixel_decoder import PixelDecoderDCT
from modules.CrossAttn import  WAMCrossAttnCustom
from einops import rearrange
from modules.UpSample import  UNetFreqNoChannelChange


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # print("x", x.get_device())
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         # print("avg_out", avg_out.get_device())
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#
#         w = self.sigmoid(out)
#         # print("w[0:10]:", w.squeeze(-1).squeeze(-1).squeeze(-1)[0,0:10])
#         return w




class Extractor(nn.Module):
    """
    Abstract class for watermark detection.
    """
    def __init__(self) -> None:
        super(Extractor, self).__init__()

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The predicted masks and/or messages.
        """
        return ...
# TODO 1.频域嵌入，空间域提取 pass
#      2.频域上采样 完成
#      3.是否直接限制或者第一阶段使用JND 先使用感知损失试一下
#      4.
class DCTSegmentationExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """
    def __init__(
        self,
        img_decoder: CVTDecoder,
        pixel_decoder: PixelDecoderDCT,
    ) -> None:
        super(DCTSegmentationExtractor, self).__init__()
        # self.image_encoder = image_encoder
        self.img_decoder = img_decoder
        self.pixel_decoder = pixel_decoder
        self.upsample = UNetFreqNoChannelChange()
        self.wam_cross = WAMCrossAttnCustom(nbits=32)

    def forward(
        self,
        imgs: torch.Tensor,
        msg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        colordctTrans = ColorDCTTransform(device=imgs.device)
        image_ycbcr = colordctTrans.rgb2ycbcr_batch(imgs) / 255.0
        # image_ycbcr.shape = (B, 3, H, W)

        # 3) 提取 Y 通道 (B, H, W)，对其做 DCT
        y_chan = image_ycbcr[:, 0, :, :]  # shape: (B, H, W)
        y_dct = colordctTrans.dct_dxc(y_chan)  # shape: (B, 64, H/8, W/8)



        img_embedding = self.img_decoder(y_dct)
        # 添加一个crossattn
        cross_massage = self.wam_cross(img_embedding,msg_embeddings)
        rev_cross_massage = self.upsample(cross_massage)
        rev_img_embedding = self.upsample(img_embedding)
        # 方案2 idct
        masks = self.pixel_decoder(rev_img_embedding, rev_cross_massage)

        return masks
# class SegmentationExtractor(Extractor):
#     """
#     Detects the watermark in an image as a segmentation mask + a message.
#     """
#     def __init__(
#         self,
#         image_encoder: ImageEncoderViT,
#         pixel_decoder: PixelDecoder,
#     ) -> None:
#         super(SegmentationExtractor, self).__init__()
#         self.image_encoder = image_encoder
#         self.pixel_decoder = pixel_decoder
#
#     def forward(
#         self,
#         imgs: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             imgs: (torch.Tensor) Batched images with shape BxCxHxW
#         Returns:
#             masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
#         """
#         latents = self.image_encoder(imgs)
#         masks = self.pixel_decoder(latents)
#
#         return masks


def build_extractor(name, cfg, img_size, nbits):
    if name.startswith('DCT'):
        cfg.img_decoder.image_size = int(img_size/8)
        cfg.img_decoder.in_channels = 64
        cfg.pixel_decoder.nbits = nbits
        img_decoder = CVTDecoder(**cfg.img_decoder)
        pixel_decoder = PixelDecoderDCT(**cfg.pixel_decoder)
        extractor = DCTSegmentationExtractor(img_decoder = img_decoder, pixel_decoder=pixel_decoder)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return extractor
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs = torch.randn(16, 3, 256, 256) * 2 - 1
    imgs = imgs.to(device)# 类似于图中的范围 (-2.1179 ~ 2.1179)


    extractor_config = '/home/wh/ywy/watermark-anything-main/configs/extractor_dct.yaml'
    extractor_cfg = omegaconf.OmegaConf.load(extractor_config)
    extractor_model = extractor_cfg.model
    extractor_params = extractor_cfg[extractor_model]
    extractor = build_extractor(extractor_model, extractor_params, 256, 32)
    extractor.to(device)
    output = extractor(imgs)
    print(output.shape)
