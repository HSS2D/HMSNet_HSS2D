import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class ZCurve:
    """
    Z-order curve (Morton curve) for mapping 2D points to 1D.
    This function generates a Z-curve index for a 2D grid of points.
    """
    @staticmethod
    def point_from_distance(d):
        # Converts a 1D distance to a (x, y) coordinate using Z-order curve
        x, y = 0, 0
        for i in range(32):  # Assuming 32-bit precision for Morton code
            x |= ((d >> (2 * i)) & 1) << i
            y |= ((d >> (2 * i + 1)) & 1) << i
        return x, y

    @staticmethod
    def distance_from_point(x, y):
        # Converts (x, y) coordinates to a 1D distance using Z-order curve
        d = 0
        for i in range(32):  # Assuming 32-bit precision
            d |= (x & 1) << (2 * i)
            d |= (y & 1) << (2 * i + 1)
            x >>= 1
            y >>= 1
        return d

class HSS(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            sizes=[32, 64, 128, 256, 512]  # 支持的图像尺寸
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.x_proj_weight = self.x_proj.weight.unsqueeze(0).to("cuda")
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        )
        self.dt_projs_weight = self.dt_projs.weight.unsqueeze(0).to("cuda")
        self.dt_projs_bias = self.dt_projs.bias.unsqueeze(0).to("cuda")
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.forward_core = self.forward_HSS
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # 预计算不同尺寸的Z曲线索引和逆索引
        self.register_zcurve_indices(sizes)

    def register_zcurve_indices(self, sizes):
        for size in sizes:
            H = W = size
            L = H * W
            zcurve_indices_1d = [x * W + y for (x, y) in [ZCurve.point_from_distance(i) for i in range(L)]]
            zcurve_indices_1d = torch.tensor(zcurve_indices_1d, dtype=torch.long)
            buffer_name = f'zcurve_indices_{size}x{size}_1d'
            self.register_buffer(buffer_name, zcurve_indices_1d)

            # 生成逆索引
            inverse_zcurve_indices_1d = torch.zeros(L, dtype=torch.long)
            inverse_zcurve_indices_1d[zcurve_indices_1d] = torch.arange(L)
            inverse_buffer_name = f'inverse_zcurve_indices_{size}x{size}_1d'
            self.register_buffer(inverse_buffer_name, inverse_zcurve_indices_1d)

    @staticmethod
    def dt_init(
                dt_rank,
                d_inner,
                dt_scale=1.0,
                dt_init="random",
                dt_min=0.001,
                dt_max=0.1,
                dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_HSS(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 1

        # 根据输入尺寸选择对应的Z曲线索引
        if L == 32 * 32:
            zcurve_indices = self.zcurve_indices_32x32_1d
            inverse_zcurve_indices = self.inverse_zcurve_indices_32x32_1d
        elif L == 64 * 64:
            zcurve_indices = self.zcurve_indices_64x64_1d
            inverse_zcurve_indices = self.inverse_zcurve_indices_64x64_1d
        elif L == 128 * 128:
            zcurve_indices = self.zcurve_indices_128x128_1d
            inverse_zcurve_indices = self.inverse_zcurve_indices_128x128_1d
        elif L == 256 * 256:
            zcurve_indices = self.zcurve_indices_256x256_1d
            inverse_zcurve_indices = self.inverse_zcurve_indices_256x256_1d
        elif L == 512 * 512:
            zcurve_indices = self.zcurve_indices_512x512_1d
            inverse_zcurve_indices = self.inverse_zcurve_indices_512x512_1d
        else:
            raise ValueError(f"Unsupported image size: {H}x{W}")

        # 使用预计算的Z曲线索引
        xs = x.view(B, -1, L).unsqueeze(1).to(x.device)
        xs = xs[:, :, :, zcurve_indices].contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 调用选择性扫描机制
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y.dtype == torch.float

        return out_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        L = H * W

        # 投影层
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # 前向传播核心
        y = self.forward_core(x)
        assert y.dtype == torch.float32

        # 根据输入尺寸选择逆Z曲线索引
        if L == 32 * 32:
            inverse_zcurve_indices = self.inverse_zcurve_indices_32x32_1d
        elif L == 64 * 64:
            inverse_zcurve_indices = self.inverse_zcurve_indices_64x64_1d
        elif L == 128 * 128:
            inverse_zcurve_indices = self.inverse_zcurve_indices_128x128_1d
        elif L == 256 * 256:
            inverse_zcurve_indices = self.inverse_zcurve_indices_256x256_1d
        elif L == 512 * 512:
            inverse_zcurve_indices = self.inverse_zcurve_indices_512x512_1d
        else:
            raise ValueError(f"Unsupported image size: {H}x{W}")

        # 使用预计算的逆Z曲线索引恢复结构
        y = y[:, :, inverse_zcurve_indices].contiguous()
        y = y.view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out
