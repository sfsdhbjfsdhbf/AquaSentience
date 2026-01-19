import math
import torch
import torch.nn as nn

class WAMCrossAttnCustom(nn.Module):
    def __init__(self, nbits: int):
        """
        :param nbits:       水印比特数 => Embedding大小 (2*nbits, hidden_size)
        :param hidden_size: 每行 Embedding 的维度
        :param n:           要将 2*nbits 映射到的维度
        """
        super().__init__()
        # (1) 消息表 => (2*nbits, hidden_size)

        # (2) 两个线性层，将 (2*nbits) -> n
        #     注意: PyTorch 的 nn.Linear 会对“最后一维”操作
        #     因此我们会先 transpose，再做线性
        self.K_lin = nn.Linear(in_features=2*nbits, out_features=nbits)
        self.V_lin = nn.Linear(in_features=2*nbits, out_features=nbits)

    def forward(self,
                latents: torch.Tensor,
                msg_embeddings: torch.Tensor,
                ) -> torch.Tensor:
        """
        :param latents: 形状 (b,hidden_size, h,w), 作为 Q
        :param h, w:    空间分辨率，用于 repeat

        最终做完注意力后输出维度仍 (hidden_size, h*w).
        """
        b,c,h,w = latents.shape
        latents = latents.reshape(b, c, -1)
        # ==========================
        # (A) 处理消息表 E
        # ==========================
        E = msg_embeddings.weight      # => (2*nbits, hidden_size)
        E_t = E.transpose(0, 1)                 # => (hidden_size, 2*nbits)

        # ---- K ----
        # 线性映射: (hidden_size, 2*nbits) -> (hidden_size, n)
        K_2d = self.K_lin(E_t)                  # => (hidden_size, n)
        # 在 n 维上取 max => (hidden_size,)
        K_1d = K_2d.sum(dim=1)               # => (hidden_size,)
        # repeat 到 (hidden_size, h*w)
        K = K_1d.unsqueeze(-1).repeat(1, h*w)   # => (hidden_size, h*w)

        # ---- V ----
        V_2d = self.V_lin(E_t)                  # => (hidden_size, n)
        V_1d = V_2d.sum(dim=1)               # => (hidden_size,)
        V = V_1d.unsqueeze(-1).repeat(1, h*w)   # => (hidden_size, h*w)

        # ==========================
        # (B) Q 就是 (hidden_size, h*w)
        # ==========================
        Q = latents  # => (hidden_size, h*w)

        # ==========================
        # (C) 在 "hidden_size" 维上做注意力
        #     => Q,K,V 皆 (hidden_size, h*w)
        # ==========================
        # 公式: Attn(Q,K,V) = softmax( (Q @ K^T)/sqrt(d) ) * V
        # 其中 d = h*w (embedding维度)

        d = Q.shape[1]   # = h*w

        # Q@K^T => (hidden_size, hidden_size)
        # 先 K 转置 (h*w, hidden_size)
        K_t = K.transpose(0, 1)                # => (h*w, hidden_size)
        attn_scores = (Q @ K_t) / math.sqrt(d) # => (hidden_size, hidden_size)

        attn_probs = torch.softmax(attn_scores, dim=-1)  # => (hidden_size, hidden_size)

        # 最终乘 V => (hidden_size, hidden_size) x (hidden_size, h*w) = (hidden_size, h*w)
                    # => (h*w, hidden_size)
        out = attn_probs @ V               # => (hidden_size, h*w)
        out = out.reshape(b,c,h,w)
        # 返回跟 Q 同形状
        return out


if __name__ == "__main__":
    nbits = 4
    hidden_size = 16
    n = 10
    h, w = 4, 4   # => h*w=16

    model = WAMCrossAttnCustom(nbits, hidden_size, n)

    # latents => (hidden_size, h*w) => (16,16)
    latents = torch.randn(hidden_size, h*w)
    out_feats = model(latents, h, w)
    print("out_feats shape:", out_feats.shape)
    # => (16,16)
