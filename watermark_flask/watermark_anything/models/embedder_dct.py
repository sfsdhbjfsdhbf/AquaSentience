# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import omegaconf
import torch
from torch import nn
import sys
import os

sys.path.append(os.path.abspath('/home/wh/ywy/watermark-anything-main/watermark_anything'))

from modules.vae import VAEEncoder, VAEDecoder
from modules.msg_processor import MsgProcessor
from modules.Ltransformer import CVTEncoder
from modules.dct_ycbcr import ColorDCTTransform

class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """

    def __init__(self) -> None:
        super(Embedder, self).__init__()

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        """
        Generate a random message
        """
        return ...

    def get_last_layer(self) -> torch.Tensor:
        return None

    def forward(
            self,
            imgs: torch.Tensor,
            msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        return ...

class DCTEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
            self,
            encoder: CVTEncoder,
            msg_processor: MsgProcessor,
            final_in_channels,
    ) -> None:
        super(DCTEmbedder, self).__init__()
        self.encoder = encoder
        self.msg_processor = msg_processor
        self.final_layer = nn.Conv2d(final_in_channels, 64, kernel_size=1)
    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.final_layer.weight
        return last_layer

    def forward(
            self,
            imgs: torch.Tensor,
            msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        # 1) 构造封装了各种变换的类，指定运行设备为 imgs 所在设备
        colordctTrans = ColorDCTTransform(device=imgs.device)

        # 2) 将 RGB 批量转换为 YCbCr，然后归一化到 0~1 左右
        image_ycbcr = colordctTrans.rgb2ycbcr_batch(imgs) / 255.0
        # image_ycbcr.shape = (B, 3, H, W)

        # 3) 提取 Y 通道 (B, H, W)，对其做 DCT
        y_chan = image_ycbcr[:, 0, :, :]  # shape: (B, H, W)
        y_dct = colordctTrans.dct_dxc(y_chan)  # shape: (B, 64, H/8, W/8)

        # 4) 在 Y 通道的 DCT 系数里嵌入消息并经过后续网络处理
        latents_w = self.msg_processor(y_dct, msgs)
        im_w1 = self.encoder(latents_w)
        im_w1 = self.final_layer(im_w1)
        # im_w1.shape = (B, 64, H/8, W/8)

        # 5) 对处理后的 Y 通道做 iDCT 恢复出 (B, H, W)
        y_idct = colordctTrans.idct_dxc(im_w1)  # (B, H, W)

        # 6) 把新 Y 通道和原来的 Cb / Cr 通道拼回，得到新的 YCbCr
        #    原先的 image_ycbcr[:,1,:,:] 和 image_ycbcr[:,2,:,:] 是 Cb,Cr
        image_dct_ycbcr = torch.cat([
            y_idct.unsqueeze(1),
            image_ycbcr[:, 1, :, :].unsqueeze(1),
            image_ycbcr[:, 2, :, :].unsqueeze(1)
        ], dim=1) * 255.0
        # image_dct_ycbcr.shape = (B, 3, H, W)

        # 7) 将 YCbCr 转回 RGB
        imgs_w = colordctTrans.ycbcr2rgb_batch(image_dct_ycbcr)
        # imgs_w.shape = (B, 3, H, W)

        return imgs_w

class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
            self,
            encoder: VAEEncoder,
            decoder: VAEDecoder,
            msg_processor: MsgProcessor
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.decoder.conv_out.weight
        return last_layer

    def forward(
            self,
            imgs: torch.Tensor,
            msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        latents = self.encoder(imgs)
        latents_w = self.msg_processor(latents, msgs)
        imgs_w = self.decoder(latents_w)
        return imgs_w


def build_embedder(name, cfg, nbits,img_size):
    if name.startswith('DCT'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits * 2
        cfg.encoder.in_channels = nbits * 2 + 64
        cfg.encoder.length = nbits * 2
        cfg.encoder.image_size = int(img_size/8)

        # build the encoder, decoder and msg processor
        encoder = CVTEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        embedder = DCTEmbedder(encoder, msg_processor, cfg.encoder.in_channels)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return embedder
def build_msg_processor(msg_processor_cfg):

    msg_processor = MsgProcessor(**cfg.msg_processor)
    return msg_processor



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs = torch.randn(16, 3, 256, 256) * 2 - 1
    imgs = imgs.to(device)# 类似于图中的范围 (-2.1179 ~ 2.1179)

    # 模拟 msgs (16, 32)
    msgs = torch.randint(0, 2, (16, 32), dtype=torch.float32)  # 二值信息（0和1）
    msgs = msgs.to(device)
    embedder_config = '/home/wh/ywy/watermark-anything-main/configs/embedder_dct.yaml'
    embedder_cfg = omegaconf.OmegaConf.load(embedder_config)
    embedder_model = embedder_cfg.model
    embedder_params = embedder_cfg[embedder_model]
    embedder = build_embedder(embedder_model, embedder_params, 32,img_size=256)
    embedder = embedder.to(device)
    output = embedder(imgs,msgs)
    print(output.shape)


