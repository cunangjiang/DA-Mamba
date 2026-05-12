import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

# 4方向正交扫描STSS2D+MoE+Top2 Sparse Orthogonal Scanning Block
class STSS2D(nn.Module): # SOS Block
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
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        is_cross=False,
        router_jitter_noise=0.01,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 4 orthogonal scan directions (experts)
        self.K = 4 
        
        # --- 核心修改 1: 缓存索引时设备兼容性 ---
        self.scan_indices_cache = {}
        self.inv_indices_list = None 

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 现在是: 4 个独立的卷积，让每个专家看到的特征不同
        self.expert_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner, # 保持 Depthwise
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                bias=conv_bias,
                **factory_kwargs
            ) for _ in range(self.K)
        ])
        self.act = nn.SiLU()

        # --- K=4 Expert Parameter Initialization (保持不变) ---
        if is_cross:
            self.style_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs),)*self.K
            self.x_proj = (nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),)*self.K
            self.style_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.style_proj], dim=0))
            del self.style_proj
        else:
            self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),)*self.K
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),)*self.K
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        # --- MoE Gating Network (Top-k=2) ---
        self.top_k = 2
        
        # 增加一个 3x3 DW Conv 获取上下文信息
        self.gate_conv_context = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=False),
            nn.SiLU()
        )
        # 使用一个 1x1 卷积在空间维度上独立计算每个位置的 K 个 logits
        self.gate_proj = nn.Conv2d(self.d_inner, self.K, kernel_size=1, bias=False)
        self.router_jitter_noise = router_jitter_noise
        # ------------------------------------

        # --- 核心修改 4: 核心层命名 ---
        self.expert_forward = self._expert_forward_scan # 重命名以强调 MoE 专家角色
        self.in_style_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        # 使用较大的标准差 (std=0.1 或 0.5) 进行正态分布初始化
        nn.init.normal_(self.gate_proj.weight, mean=0, std=0.1)
        
        # --- 关键: 零初始化最后的输出层 ---
        nn.init.constant_(self.out_proj.weight, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
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

    # ------------------------------------------------------------------
    # --- 核心修改 1: 索引生成器 (包含设备兼容性修复) ---
    # ------------------------------------------------------------------
    def get_4dir_indices_cached(self, H, W, device):
        """
        Generates and caches the forward and inverse indices for the 4 orthogonal scans.
        Fix: Use str(device) for hash key compatibility.
        """
        cache_key = (H, W, str(device))
        if cache_key in self.scan_indices_cache:
            return self.scan_indices_cache[cache_key]

        L = H * W
        indices_list = []
        inv_indices_list = []

        # 0: row-major (L-to-R, T-to-B) - Standard view(L)
        idx_row = torch.arange(L, device=device)
        indices_list.append(idx_row)
        inv_indices_list.append(torch.argsort(idx_row))

        # 1: col-major (T-to-B, L-to-R) - View(H, W) -> transpose -> view(L)
        idx_col = rearrange(torch.arange(L, device=device).view(H, W), 'h w -> w h').contiguous().view(-1)
        indices_list.append(idx_col)
        inv_indices_list.append(torch.argsort(idx_col))

        # 2: row-major, sequence reversed - flip(row)
        idx_row_seq_flip = torch.flip(idx_row, dims=[0])
        indices_list.append(idx_row_seq_flip)
        inv_indices_list.append(torch.argsort(idx_row_seq_flip))

        # 3: col-major, sequence reversed - flip(col)
        idx_col_seq_flip = torch.flip(idx_col, dims=[0])
        indices_list.append(idx_col_seq_flip)
        inv_indices_list.append(torch.argsort(idx_col_seq_flip))

        self.scan_indices_cache[cache_key] = (indices_list, inv_indices_list)
        return indices_list, inv_indices_list

    # ------------------------------------------------------------------
    # --- 核心修改 4: 重命名 forward_core -> _expert_forward_scan ---
    # ------------------------------------------------------------------
    def _expert_forward_scan(self, x_stack: torch.Tensor, style=None):
        # x_stack: (B, K, C, H, W) - 这里的 K 个 slice 已经经过了不同的卷积
        self.selective_scan = selective_scan_fn 
        B, K, C, H, W = x_stack.shape
        L = H * W
        assert K == self.K

        indices_list, inv_indices_list = self.get_4dir_indices_cached(H, W, x_stack.device)
        self.inv_indices_list = inv_indices_list
        
        # Flatten spatial: (B, K, C, L)
        x_flat_stack = rearrange(x_stack, 'b k c h w -> b k c (h w)')
        
        # --- 这里的逻辑变了 ---
        # 我们不再是把同一个 x 切成 4 份
        # 而是取 Expert 0 的输出做 Scan 0，Expert 1 的输出做 Scan 1...
        scans = []
        for k in range(K):
            # 取第 k 个专家的卷积输出 (B, C, L)
            x_k = x_flat_stack[:, k, :, :] 
            # 按第 k 个方向进行索引重排
            idx = indices_list[k]
            scans.append(x_k.index_select(2, idx))
            
        xs = torch.stack(scans, dim=1) # (B, 4, C, L)
        
        # 后续 SSM 计算保持不变
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        if style is not None:
            style_flat = rearrange(style, 'b c h w -> b c (h w)')
            style_scans = []
            for indices in indices_list:
                style_scans.append(style_flat.index_select(2, indices))
            style_xs = torch.stack(style_scans, dim=1)
            s_dbl = torch.einsum("b k d l, k c d -> b k c l", style_xs.view(B, K, -1, L), self.style_proj_weight)
            dts, Bs = torch.split(s_dbl, [self.dt_rank, self.d_state], dim=2)
            Cs = x_dbl
        else:
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        if style is not None:
            style = style_xs.float().view(B, -1, L)
            out_y = selective_scan_fn(
                style, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
            ).view(B, K, -1, L)
        else:
            out_y = selective_scan_fn(
                xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
            ).view(B, K, -1, L)
            
        return out_y

    def forward(self, x: torch.Tensor, style=None, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        
        x, z = xz.chunk(2, dim=-1)
        if style is not None:
            style = self.in_style_proj(style)
            
        x_conv_in = x.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        
        # x_conv = self.act(self.conv2d(x_conv))
        # --- 计算 K 个独立的卷积 ---
        expert_outputs_list = []
        for k in range(self.K):
            # 每个专家用自己的卷积核处理输入
            out_k = self.act(self.expert_convs[k](x_conv_in))
            expert_outputs_list.append(out_k)
        
        # 堆叠: (B, K, C, H, W)
        x_conv_stack = torch.stack(expert_outputs_list, dim=1)
        
        # 共享的 x_conv 用于门控输入 (这里我们取所有专家输出的平均，或者依然用独立卷积？)
        # 更好的做法：用原始 x_conv_in 走门控，但加上 3x3 上下文
        
        style_conv = None
        if style is not None:
            style_conv = style.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
            # style 也可以做类似处理，但为了节省显存，这里仅做一次 shared conv 吧
            # 如果显存允许，也可以做 K 个 conv
            # 这里暂且复用 self.expert_convs[0] 或者维持原样
            # 为了简单，我们假设 style 主要影响 SSM 参数，不参与 x_conv 差异化
            # 但代码里需要 style_conv，这里简单处理：
            style_conv = self.act(self.expert_convs[0](style_conv)) # 仅示例
        
        # ------------------------------------------------------------------
        # --- Hard Top-2 MoE Gating with STE ---
        # 1. MoE Gating
        # 1. 上下文提取 (3x3)
        gate_context = self.gate_conv_context(x_conv_in) # (B, C, H, W)
        # 2. 投影 (1x1)
        gate_logits = self.gate_proj(gate_context) # (B, K, H, W)
        
        # <--- 核心修改: 2. 仅在训练时添加抖动噪声 ---
        if self.training and self.router_jitter_noise > 0:
            # randn_like 会自动匹配 device 和 dtype
            noise = torch.randn_like(gate_logits) * self.router_jitter_noise
            gate_logits = gate_logits + noise
        # ----------------------------------------------
        
        # 2. Compute "Soft" weights (for backward pass)
        gate_weights_softmax = F.softmax(gate_logits, dim=1) # (B, K=4)

        # 3. Select Top-k (k=2) indices
        top_k_weights, top_k_indices = torch.topk(gate_weights_softmax, self.top_k, dim=1) # (B, 2)
        
        # 4. Normalize Top-k weights (Sum to 1 for forward)
        top_k_weights_norm = top_k_weights / (top_k_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # --- 核心修改 3: 权重归一化一致性 ---
        # 应用 (K/k) 缩放因子 (4/2 = 2) 来补偿被丢弃的专家的贡献，平衡梯度范数
        scaling_factor = self.K / self.top_k 
        top_k_weights_norm = top_k_weights_norm * scaling_factor

        # 5. Create "Hard" sparse weights (for forward pass)
        sparse_gate_weights = torch.zeros_like(gate_weights_softmax)
        sparse_gate_weights.scatter_(1, top_k_indices, top_k_weights_norm) # [B, K, H, W]
        
        # 6. Apply Straight-Through Estimator (STE)
        gate_weights = gate_weights_softmax + (sparse_gate_weights - gate_weights_softmax).detach()
        
        # ------------------------------------------------------------------

        # 6. Compute all K=4 expert outputs (DENSE calculation)
        # out_y_scans = self._expert_forward_scan(x_conv, style_conv) # (B, 4, C_inner, L)
        out_y_scans = self._expert_forward_scan(x_conv_stack, style_conv)
        B_y, K_y, C_y, L_y = out_y_scans.shape

        # --- 7. (新) 将所有 K 个专家输出 "逆向扫描" 回原始空间顺序 ---
        # self.inv_indices_list 是在 _expert_forward_scan 中计算和缓存的 [B, K, C, L]
        inv_idx = torch.stack(self.inv_indices_list, dim=0) # (K, L)
        inv_exp = inv_idx.view(1, K_y, 1, L_y).expand(B_y, -1, C_y, -1) # (B, K, C, L)

        # 使用 gather 将所有 K 个专家的 L 维数据恢复到原始空间顺序 (但仍是扁平的 L)
        out_y_spatial_flat = torch.gather(out_y_scans, dim=3, index=inv_exp) # (B, K, C, L)

        # 恢复到 (B, K, C, H, W)
        out_y_spatial = rearrange(out_y_spatial_flat, 'b k c (h w) -> b k c h w', h=H, w=W)

        # --- 8. (新) 应用逐令牌 (Per-Token) 权重 ---
        # gate_weights 形状为 [B, K, H, W]
        # 将 gate_weights 扩展为 [B, K, 1, H, W] 以便广播
        gate_weights_expanded = gate_weights.unsqueeze(2) 

        # (B, K, C, H, W) * (B, K, 1, H, W) -> (B, K, C, H, W)
        out_y_weighted = out_y_spatial * gate_weights_expanded

        # --- 9. (新) 聚合 (Sum) ---
        # 沿 K 维度求和 (dim=1)
        y_agg = out_y_weighted.sum(dim=1) # [B, C_inner, H, W]

        # --- 10. (新) Reshape 并继续 ---
        # 恢复到 (B, H, W, C_inner)
        y = y_agg.permute(0, 2, 3, 1).contiguous()
        
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.training:
            return out, gate_logits
        else:
            return out, gate_weights_softmax

   
class STVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        is_cross: bool = False,
        router_jitter_noise: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = STSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, is_cross=is_cross, router_jitter_noise=router_jitter_noise, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor, style=None):
        # x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        if style is not None:
            rnd = torch.rand(style.shape[1])
            indexes = torch.argsort(rnd)
            style = style[:,indexes,:]
            
        # --- 修改: 处理 STSS2D 在 train 模式下的 tuple 输出 ---
        attn_input = self.ln_1(input)
        
        attn_output, gate_logits = self.self_attention(attn_input, style=style)
        x = input + self.drop_path(attn_output)
        return x, gate_logits # 传递 logits


class STVSSLayer(nn.Module): 
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        is_cross=False,
        router_jitter_noise=0.01,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            STVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                is_cross=is_cross,
                router_jitter_noise=router_jitter_noise,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, style=None):
        all_aux_data = []
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x, aux = checkpoint.checkpoint(blk, x, style=style)
                if aux is not None:
                     all_aux_data.append(aux)
            else:
                x, aux = blk(x, style=style)
                if aux is not None:
                     all_aux_data.append(aux)

        if self.downsample is not None:
            x = self.downsample(x)

        # 始终返回 x 和最后一个 block 的 aux_data
        # (aux 在训练时是 logits, 在评估时是 weights)
        return x, all_aux_data


    


