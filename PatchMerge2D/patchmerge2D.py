import torch
import torch.nn as nn
from einops import rearrange


# 下采样操作，特征图的空间维度减半，同时通道数目翻倍
class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # 输入特征的维度
        self.dim = dim

        # 对 4 * dim 通道的特征进行归一化处理
        self.norm = norm_layer(4 * dim)

        # 在下采样过程中，PatchMerging 将 2x2 区域的像素拼接成一个新的像素
        # 因此通道数在拼接后是原来的4倍，即 4 * dim，通过 nn.Linear(4 * dim, 2 * dim) 将通道数从 4 * dim 降到 2 * dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            # 如果宽度或高度不是偶数（即出现形状不匹配的情况），则将 SHAPE_FIX 设置为下采样后的形状 [H // 2, W // 2]
            SHAPE_FIX[0] = (H // 2)
            SHAPE_FIX[1] = (W // 2)

        # x0 提取了偶数行和偶数列的元素；
        # x1 提取了奇数行和偶数列的元素；
        # x2 提取了偶数行和奇数列的元素；
        # x3 提取了奇数行和奇数列的元素
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # 处理输入张量的高度 H 或宽度 W 不是偶数的情况
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        # 通过上述四次操作，原始的 H x W 空间被分解成了四个子区域，每个子区域的大小为 H/2 x W/2
        # 然后可以进一步将它们在通道维度进行拼接（如 torch.cat）成一个下采样的张量。
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

# 上采样操作，空间尺寸x2，通道数减半
# 提供了一种可学习的上采样方式，通过将通道信息重新分配到空间维度
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        # rearrange 将通道维度的部分信息分配到 h 和 w（即 H 和 W）维度上，增加空间分辨率。
        # 例如，输入是 B, H, W, C 的张量，经过这个操作后
        # 变为 B, H*dim_scale, W*dim_scale, C // dim_scale
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        return x

# Input shape: torch.Size([6, 12, 12, 192])
# Output shape: torch.Size([6, 6, 6, 384])
# Input shape: torch.Size([6, 6, 6, 384])
# Output shape: torch.Size([6, 12, 12, 192])