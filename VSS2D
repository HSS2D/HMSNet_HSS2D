import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    pass

# 尝试导入另一个版本的 selective_scan 实现，作为备选方案
# 这为代码提供了更好的兼容性
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except ImportError:
    selective_scan_fn_v1 = None
    pass


class SS2D(nn.Module):
    """
    2D 状态空间模型 (SS2D) 模块。
    此模块将 Mamba/S4 的思想应用于 2D 数据，如图像。
    它通过在四个方向（上->下、左->右 及其相反方向）对图像进行扫描，
    从而捕捉长距离依赖关系。
    """
    def __init__(
        self,
        d_model,                # 输入特征维度
        d_state=16,             # 状态向量 A 和 B 的维度 N
        d_conv=3,               # 卷积核大小
        expand=2,               # 隐藏层扩展倍数 E
        dt_rank="auto",         # 时间步 dt 的秩
        dt_min=0.001,           # 时间步 dt 的最小值
        dt_max=0.1,             # 时间步 dt 的最大值
        dt_init="random",       # dt 投影权重的初始化方法 ("random" or "constant")
        dt_scale=1.0,           # dt 投影权重的缩放因子
        dt_init_floor=1e-4,     # dt 的最小初始化值
        dropout=0.,             # Dropout 概率
        conv_bias=True,         # 卷积层是否使用偏置
        bias=False,             # 线性层是否使用偏置
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 扩展后的内部维度 D = E * d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # --- 输入线性投影 ---
        # 将输入从 d_model 投影到 2 * d_inner，用于后续分割为 x 和 z (门控)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # --- 卷积分支 ---
        # 深度可分离卷积，用于捕捉局部信息
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 深度可分离卷积
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,  # 保持分辨率不变
            **factory_kwargs,
        )
        self.act = nn.SiLU()  # 激活函数

        # --- SSM 参数投影 ---
        # x_proj 用于从输入 x 生成 SSM 的参数 (dt, B, C)
        # K=4 代表四个扫描方向
        self.x_proj_weight = nn.Parameter(torch.randn(4, self.dt_rank + self.d_state * 2, self.d_inner, **factory_kwargs))

        # dt_projs 用于从 dt_rank 维度的向量生成 d_inner 维度的 dt
        dt_projs_list = [self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projs_list], dim=0)) # (K=4, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projs_list], dim=0))   # (K=4, d_inner)

        # --- SSM 核心参数 A 和 D ---
        # A_logs 是状态矩阵 A 的对数值，D 是跳跃连接参数
        # K=4 代表四个扫描方向
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K * d_inner, d_state)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K * d_inner)

        # 选择前向核心函数，默认为 v0
        self.forward_core = self.forward_corev0

        # --- 输出处理 ---
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """初始化 dt 投影层"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 初始化权重
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化偏置，确保 softplus(bias) 的值在 [dt_min, dt_max] 范围内
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        # 计算 softplus 的逆函数，以设置偏置
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # 标记此偏置不应被重新初始化
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True, **factory_kwargs):
        """初始化状态矩阵 A 的对数值 (log A)"""
        # S4D 风格的实数初始化
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, **factory_kwargs),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 保持 A_log 为 float32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # 通常不对 A 进行权重衰减
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True, **factory_kwargs):
        """初始化跳跃连接参数 D"""
        D = torch.ones(d_inner, **factory_kwargs)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        
        D = nn.Parameter(D)  # 保持 D 为 float32
        D._no_weight_decay = True  # 通常不对 D 进行权重衰减
        return D

    def forward_corev0(self, x: torch.Tensor):
        """
        核心前向传播函数，执行四向扫描。
        """
        # 确保使用正确的 selective_scan 实现
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please install mamba-ssm or selective_scan.")
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4  # 四个扫描方向

        # 1. 准备四向扫描的输入序列
        # (b, d, h, w) -> (b, k, d, l) where k=4
        # 方向1: (h, w) - 逐行扫描
        # 方向2: (w, h) - 逐列扫描 (通过转置实现)
        x_hwwh = torch.stack([
            x.view(B, -1, L),                                       # (B, C, L)
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L) # (B, C, L)
        ], dim=1).view(B, 2, -1, L)
        
        # 方向3 & 4: 通过翻转序列得到反向扫描
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B, K, C, L)

        # 2. 计算 SSM 参数
        # 使用 einsum 高效计算，将输入 xs 投影到 (dt, B, C) 的参数空间
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        # 分割出 dt, B, C
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # 进一步投影 dt
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # 3. 准备 selective_scan 的输入
        xs = xs.contiguous().view(B, -1, L)     # (B, K*C, L)
        dts = dts.contiguous().view(B, -1, L)   # (B, K*C, L)
        Bs = Bs.contiguous().view(B, K, -1, L)  # (B, K, d_state, L)
        Cs = Cs.contiguous().view(B, K, -1, L)  # (B, K, d_state, L)
        Ds = self.Ds.view(-1)                   # (K*C)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # (K*C, d_state)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (K*C)

        # 4. 执行选择性扫描
        # 这是模型的核心计算，对四个方向的序列并行处理
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # 5. 将扫描结果转换回 2D 格式
        # (b, k, d, l) -> y1, y2, y3, y4
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 返回四个方向的扫描结果
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        模块的主前向传播函数。
        """
        # 输入形状 (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        # 1. 输入投影，并分割为 x (主路径) 和 z (门控路径)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, H, W, d_inner)

        # 2. 卷积分支
        # (B, H, W, d_inner) -> (B, d_inner, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # 3. SSM 核心计算
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        # 4. 融合四个方向的输出
        y = y1 + y2 + y3 + y4

        # 5. 结果处理和门控
        # (B, d_inner, L) -> (B, H, W, d_inner)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        
        # LayerNorm，兼容半精度训练
        y = self.out_norm(y)

        # 门控机制 (Gated Linear Unit)
        y = y * F.silu(z)

        # 6. 输出投影
        out = self.out_proj(y)
        
        # Dropout
        out = self.dropout(out)

        # (B, H, W, d_model) -> (B, d_model, H, W)
        out = out.permute(0, 3, 1, 2).contiguous()
        
        return out
