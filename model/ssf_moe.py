import os
import sys
import time
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# -----------------------------------------------------------------------------
# Flexible imports: support both
#   1) placing this file under your project/model/ as ssf_moe.py
#   2) standalone local test with vmamba_moe.py in the same folder
# -----------------------------------------------------------------------------
try:
    from model.vmamba_moe import STSS2D
except Exception:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from vmamba_moe import STSS2D

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE_AVAILABLE = True
except Exception:
    FlopCountAnalysis = None
    _FVCORE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Experts for SSF_MoE
# -----------------------------------------------------------------------------
class STSS2DExpert(nn.Module):
    """
    Expert 1: Dynamic spatial fusion expert based on STSS2D.
    """
    def __init__(self, d_model: int, d_state: int = 16, **kwargs):
        super().__init__()
        self.pre_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1, padding=0)
        self.stss_module = STSS2D(d_model=d_model, d_state=d_state, **kwargs)

    def forward(self, x_tar: torch.Tensor, x_ref: torch.Tensor):
        x_fused = torch.cat([x_tar, x_ref], dim=1)
        x_pre = self.pre_conv(x_fused)
        x_in_stss = x_pre.permute(0, 2, 3, 1).contiguous()
        y_stss, aux_data = self.stss_module(x_in_stss)
        y_corr = y_stss.permute(0, 3, 1, 2).contiguous()
        return y_corr, [aux_data] if aux_data is not None else []


class GradientModulationExpert(nn.Module):
    """
    Expert 2: Gradient-guided modulation expert.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.project = nn.Conv2d(dim * 2, dim, 1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.grad_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)

        with torch.no_grad():
            kernel = torch.tensor(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                dtype=torch.float32,
            )
            kernel = kernel.reshape(1, 1, 3, 3).repeat(dim, 1, 1, 1)
            self.grad_conv.weight.copy_(kernel)
            if self.grad_conv.bias is not None:
                self.grad_conv.bias.zero_()

    def forward(self, x_tar: torch.Tensor, x_ref: torch.Tensor):
        x = self.project(torch.cat([x_tar, x_ref], dim=1))
        feat = self.conv1(x)
        grad = torch.abs(self.grad_conv(feat))
        attention = self.gate(grad)
        out = feat * (1 + attention)
        return self.conv2(out), []


class GatedSelectionExpert(nn.Module):
    """
    Expert 3: Gated target-preserving expert.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.main_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)
        nn.init.constant_(self.out_proj.weight, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x_tar: torch.Tensor, _x_ref: torch.Tensor = None):
        feat = self.main_conv(x_tar)
        mask = self.gate_conv(x_tar)
        return self.out_proj(feat * mask), []


class DeformableAlignmentExpert(nn.Module):
    """
    Expert 4: Deformable alignment expert.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.offset_conv = nn.Conv2d(dim * 2, 18, 3, 1, 1)
        self.dcn = DeformConv2d(dim, dim, 3, padding=1)
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x_tar: torch.Tensor, x_ref: torch.Tensor):
        offset = self.offset_conv(torch.cat([x_tar, x_ref], dim=1))
        aligned_ref = self.dcn(x_ref, offset)
        return self.fusion(aligned_ref), []


# -----------------------------------------------------------------------------
# Main module: SSF_MoE (renamed from ARFU_MoE)
# -----------------------------------------------------------------------------
class SSF_MoE(nn.Module):
    """
    Semantic-aware Soft Fusion Mixture of Experts.

    Input:
        tar: [B, C, H, W]
        ref: [B, C, H, W]

    Output:
        fused: [B, C, H, W]
        aux_data: list
    """
    def __init__(self, dim: int = 96, num_experts: int = 4, d_state: int = 16, **kwargs):
        super().__init__()
        self.dim = dim
        self.K = num_experts
        if self.K != 4:
            raise ValueError(f"SSF_MoE is currently implemented with 4 experts, but got num_experts={self.K}.")

        self.pre_tar = nn.Conv2d(dim, dim, 3, 1, 1)
        self.pre_ref = nn.Conv2d(dim, dim, 3, 1, 1)

        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim * 2, self.K, 1),
        )
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)
        if self.gate_net[-1].bias is not None:
            nn.init.constant_(self.gate_net[-1].bias, 0)

        self.experts = nn.ModuleList([
            STSS2DExpert(d_model=dim, d_state=d_state, **kwargs),
            GradientModulationExpert(dim),
            GatedSelectionExpert(dim),
            DeformableAlignmentExpert(dim),
        ])

        self.post_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.constant_(self.post_conv[-1].weight, 0)
        if self.post_conv[-1].bias is not None:
            nn.init.constant_(self.post_conv[-1].bias, 0)

    def forward(self, tar: torch.Tensor, ref: torch.Tensor):
        x_tar = self.pre_tar(tar)
        x_ref = self.pre_ref(ref)
        x_diff = x_tar - x_ref

        gate_in = torch.cat([x_tar, x_ref, x_diff], dim=1)
        gate_logits = self.gate_net(gate_in)
        weights = F.softmax(gate_logits, dim=1).unsqueeze(2)  # [B, 4, 1, H, W]

        y1, aux1 = self.experts[0](x_tar, x_ref)
        y2, _ = self.experts[1](x_tar, x_ref)
        y3, _ = self.experts[2](x_tar, None)
        y4, _ = self.experts[3](x_tar, x_ref)

        y_stack = torch.stack([y1, y2, y3, y4], dim=1)
        y_agg = (y_stack * weights).sum(dim=1)

        all_aux_data = [gate_logits] if self.training else [weights.squeeze(2)]
        if self.training and aux1:
            all_aux_data.extend(aux1)

        return tar + self.post_conv(y_agg), all_aux_data


class SSFMoEWrapper(nn.Module):
    """
    Wrapper used for FLOPs/time analysis so the returned object is only a tensor.
    """
    def __init__(self, core: SSF_MoE):
        super().__init__()
        self.core = core

    def forward(self, tar: torch.Tensor, ref: torch.Tensor):
        out, _ = self.core(tar, ref)
        return out


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    tar: torch.Tensor,
    ref: torch.Tensor,
    warmup: int = 20,
    iters: int = 100,
) -> float:
    model.eval()

    for _ in range(warmup):
        _ = model(tar, ref)

    if tar.is_cuda:
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = model(tar, ref)
    if tar.is_cuda:
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000.0 / iters


def analyze_ssf_moe(
    batch_size: int = 1,
    dim: int = 96,
    height: int = 56,
    width: int = 56,
    d_state: int = 16,
    device: str = None,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)
    tar = torch.randn(batch_size, dim, height, width, device=device_obj)
    ref = torch.randn(batch_size, dim, height, width, device=device_obj)

    core_model = SSF_MoE(dim=dim, d_state=d_state).to(device_obj)
    wrapped_model = SSFMoEWrapper(core_model).to(device_obj)
    wrapped_model.eval()

    total_params, trainable_params = count_parameters(core_model)

    if _FVCORE_AVAILABLE:
        flops_analyzer = FlopCountAnalysis(wrapped_model, (tar, ref))
        total_flops = flops_analyzer.total()
    else:
        total_flops = None

    avg_time_ms = measure_inference_time(wrapped_model, tar, ref, warmup=warmup, iters=iters)

    results = {
        "device": str(device_obj),
        "input_shape_tar": tuple(tar.shape),
        "input_shape_ref": tuple(ref.shape),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_flops": total_flops,
        "avg_inference_time_ms": avg_time_ms,
    }
    return results


if __name__ == "__main__":
    # You can modify these values directly when testing.
    batch_size = 1
    dim = 96
    height = 56
    width = 56
    d_state = 16
    warmup = 20
    iters = 100

    results = analyze_ssf_moe(
        batch_size=batch_size,
        dim=dim,
        height=height,
        width=width,
        d_state=d_state,
        warmup=warmup,
        iters=iters,
    )

    print("=" * 80)
    print("SSF_MoE analysis")
    print("=" * 80)
    print(f"Device:               {results['device']}")
    print(f"Tar shape:            {results['input_shape_tar']}")
    print(f"Ref shape:            {results['input_shape_ref']}")
    print(f"Total params:         {results['total_params']:,} ({results['total_params'] / 1e6:.4f} M)")
    print(f"Trainable params:     {results['trainable_params']:,} ({results['trainable_params'] / 1e6:.4f} M)")

    if results['total_flops'] is not None:
        print(f"Total FLOPs:          {results['total_flops']:,} ({results['total_flops'] / 1e9:.4f} GFLOPs)")
    else:
        print("Total FLOPs:          fvcore is not installed, so FLOPs were not computed.")

    print(f"Inference time:       {results['avg_inference_time_ms']:.4f} ms / iter")
    print("=" * 80)
