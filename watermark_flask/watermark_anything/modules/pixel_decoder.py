# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# SAM

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import Upsample


class PixelDecoder(nn.Module):
  def __init__(
    self,
    *,
    embed_dim: int,
    nbits: int = 0,
    activation: Type[nn.Module] = nn.GELU,
    upscale_stages: List[int] = [4, 2, 2],
    upscale_type: str = 'bilinear',
    sigmoid_output: bool = False,
  ) -> None:
    """
    Predicts masks given an image embedding, using a simple CNN.

    Arguments:
      embed_dim (int): the input channel dimension
      nbits (int): the number of bits to predict (0 for zero-bit)
      activation (nn.Module): the type of activation to use when
      upscaling masks
      upscale_stages (List[int]): the upscaling factors to use
      upscale_type (str): the type of upscaling to use
      sigmoid_output (bool): whether to apply sigmoid to the output
    """
    super().__init__()
    self.embed_dim = embed_dim
    self.nbits = nbits

    self.output_upscaling = []
    for up_factor in upscale_stages:
        self.output_upscaling += [
            Upsample(upscale_type, embed_dim, embed_dim // up_factor, up_factor, activation),
        ]
        embed_dim //= up_factor
    self.output_upscaling = nn.Sequential(*self.output_upscaling)

    self.out_channels = self.nbits + 1
    self.last_layer = nn.Conv2d(embed_dim, self.out_channels, kernel_size=1, bias=True)
    self.sigmoid_output = sigmoid_output




  def forward(
    self,
    image_embeddings: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict masks given image and prompt embeddings.

    Arguments:
      image_embeddings (torch.Tensor): the embeddings from the image encoder

    Returns:
      torch.Tensor: batched predicted masks (1+nbits)
    """

    # Expand per-image data in batch direction to be per-mask
    b, c, h, w = image_embeddings.shape  # b c h/16 w/16

    # Upscale mask embeddings and predict masks using the mask tokens
    upscaled_embedding = self.output_upscaling(image_embeddings)  # b c h/16 w/16 -> b c/16 h w
    preds = self.last_layer(upscaled_embedding)  # b c/16 h w -> b 1+nbits h w
    # TODO 这个地方有bug，启动就出错
    if self.sigmoid_output:
      preds = F.sigmoid(preds)

    return preds

# TODO image_embbeding 解耦成检测头和消息头 最后在concat
class PixelDecoderDCT(nn.Module):
    def __init__(
            self,
            *,
            embed_dim: int,
            nbits: int = 0,
            sigmoid_output: bool = False,
    ) -> None:
        """
    Predicts masks given an image embedding, using a simple CNN.

    Arguments:
      embed_dim (int): the input channel dimension
      nbits (int): the number of bits to predict (0 for zero-bit)
      sigmoid_output (bool): whether to apply sigmoid to the output
    """
        super().__init__()
        self.embed_dim = embed_dim
        self.nbits = nbits

        # 分别用两个卷积头来输出 detection 结果和 message 结果
        self.detection_head = nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)
        self.message_head = nn.Conv2d(embed_dim, nbits, kernel_size=1, bias=True)

        self.sigmoid_output = sigmoid_output

    def forward(
            self,
            image_embeddings: torch.Tensor,
            cross_massage: torch.Tensor
    ) -> torch.Tensor:
        """
    Predict masks (or detections + messages) given image and prompt embeddings.

    Arguments:
      image_embeddings (torch.Tensor): the embeddings from the image encoder (B,C,H,W)
      cross_massage (torch.Tensor): the embeddings from another source (B,C,H,W)

    Returns:
      torch.Tensor: concatenated detections + messages (shape: B, 1+nbits, H, W)
    """

        b, c, h, w = image_embeddings.shape  # 与 cross_massage 相同形状

        # 1. 分别得到检测结果与消息结果
        #    detection_res: [b, 1, h, w]
        detection_res = self.detection_head(image_embeddings)

        #    message_res: [b, nbits, h, w]
        message_res = self.message_head(cross_massage)

        # 2. 在通道维度上拼接
        preds = torch.cat((detection_res, message_res), dim=1)  # [b, 1+nbits, h, w]

        # 3. 如果需要 sigmoid，则进行后处理
        if self.sigmoid_output:
            preds = F.sigmoid(preds)
        return preds

