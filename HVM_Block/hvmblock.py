from models.vm_hssnet.hss2d import HSS
import torch
import torch.nn as nn
from typing import Callable

from functools import partial

class HVMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            # 归一化操作
            # 通过将 norm_layer 作为 Callable，可以在模型定义时灵活传入不同的归一化层。
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            d_state: int = 16,
            expand:int=2,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = HSS(d_model=hidden_dim, d_state=d_state, expand=expand)

    def forward(self, input: torch.Tensor):
        x = input + self.self_attention(self.ln_1(input))
        return x

# 每个 HSSLayer 包含 HVM Block 和 1 个 PatchMergin2D
class HSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            expand,
            d_state=16,
            norm_layer=nn.LayerNorm,
            # 下采样操作，此处为 PatchMergin2D
            downsample=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.use_checkpoint = use_checkpoint
        # for 循环的作用是用来创建一个包含多个 HVM Block 实例的列表。
        # 这些 HVM Block 实例会被存储在 self.blocks 这个 nn.ModuleList 中，并在前向传播时被依次调用
        self.blocks = nn.ModuleList([
            HVMBlock(
                hidden_dim=dim,
                expand = expand,
                norm_layer=norm_layer,
                d_state=d_state,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 遍历 HVM Block
        for blk in self.blocks:
            x = blk(x)

        # 遍历完 HVM Block 后，进行下采样，尺寸减半，维度翻倍
        if self.downsample is not None:
            x = self.downsample(x)
        return x


if __name__ == '__main__':
    hss_b1 = HSSLayer(
        dim=64,
        depth=1,
        d_state=32,
        expand=2
    )
    hss_b1.cuda()
    input = torch.randn(1, 64, 256, 256).cuda()
    x = input.permute(0, 2, 3, 1)
    x_h = hss_b1(x)
    print(x_h.shape)