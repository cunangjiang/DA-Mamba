import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from torchvision.ops import DeformConv2d
# from model.module_moe import *
from module_moe import *
# from model.SSF_ablation_study import wo_SSF, wo_NIE, wo_GME, wo_GSE, wo_DAE
from fvcore.nn import FlopCountAnalysis

def make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])

class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x

class Conv2D(nn.Module):
    def __init__(self, in_chl, nf, n_blks=2, act='relu'):
        super(Conv2D, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)
        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))

        return fea_L1

# ---------------------------------------------------------------------------
# --- 语义专家 (Experts for ARFU_MoE) ---
# ---------------------------------------------------------------------------

class STSS2D_Expert(nn.Module):
    """
    专家 1: 动态空间融合 (重量级)
    功能: 处理复杂的、非局部的、需要形变对齐的融合。
    """
    def __init__(self, d_model, d_state=16, **kwargs):
        super().__init__()
        # 预融合: (B, 2C, H, W) -> (B, C, H, W)
        self.pre_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1, padding=0)
        # 核心模块: STSS2D (K=4, Hard Top-2 STE)
        # 注意: **kwargs 会传递 d_conv, expand 等参数 (如果 ARFU_MoE 接收了它们)
        self.stss_module = STSS2D(d_model=d_model, d_state=d_state, **kwargs)

    def forward(self, x_tar, x_ref):
        # 1. 融合输入
        x_fused = torch.cat([x_tar, x_ref], dim=1)
        x_pre = self.pre_conv(x_fused) # (B, C, H, W)
        
        # 2. 转换为 STSS2D 期望的 (B, H, W, C)
        x_in_stss = x_pre.permute(0, 2, 3, 1).contiguous()
        
        # 3. 运行 STSS2D
        y_stss, aux_data = self.stss_module(x_in_stss)
        
        # 4. 转换回 (B, C, H, W)
        y_corr = y_stss.permute(0, 3, 1, 2).contiguous()
        
        return y_corr, [aux_data] if aux_data is not None else []

# ---------------------------------------------------------------------------
# --- E2: Gradient-Guided Modulation Expert ---
# ---------------------------------------------------------------------------
class GradientModulationExpert(nn.Module):
    """
    E2: 梯度引导调制专家
    替代普通的 RCAB。利用 Sobel 算子提取显式梯度，引导模型关注高频边缘。
    Story: "Edge-aware refinement via explicit gradient guidance."
    """
    def __init__(self, dim):
        super().__init__()
        # 降维处理
        self.project = nn.Conv2d(dim * 2, dim, 1) # 接收 cat(tar, ref)
        
        # 特征提取
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # 梯度分支 (Sobel priors)
        # 不学习的 Sobel 核，或者可学习的边缘提取器
        self.grad_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim) 
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # 初始化 Sobel 近似 (突出中心与周围的差异) 强行把卷积核初始化为“边缘检测器”，在训练刚开始时，网络就已经具备了“看清边缘”的能力。
        # 这是一种 “归纳偏置” (Inductive Bias) 的注入，能加速收敛并引导网络关注结构信息。
        with torch.no_grad():
            # 简单的拉普拉斯/边缘初始化
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            kernel = kernel.reshape(1, 1, 3, 3).repeat(dim, 1, 1, 1)
            self.grad_conv.weight.copy_(kernel)
            self.grad_conv.bias.zero_()

    def forward(self, x_tar, x_ref):
        # 1. 融合输入 (修正了您指出的问题)
        x = self.project(torch.cat([x_tar, x_ref], dim=1))
        
        # 2. 提取特征
        feat = self.conv1(x)
        
        # 3. 计算梯度图 (作为 Attention)
        grad = torch.abs(self.grad_conv(feat)) # 边缘强度
        attention = self.gate(grad) # 将梯度转化为权重
        
        # 4. 调制：在边缘处增强特征响应
        out = feat * (1 + attention) 
        
        return self.conv2(out), []

# ---------------------------------------------------------------------------
# --- E3: Gated Selection Expert (抗伪影) ---
# ---------------------------------------------------------------------------
class GatedSelectionExpert(nn.Module):
    """
    E3: 智能门控专家
    优势: 在数据增强(伪影)下，能够软性抑制噪声区域，优于 Identity。
    """
    def __init__(self, dim):
        super().__init__()
        self.main_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid()
        )
        # 零初始化最后输出，利于学习透传
        self.out_proj = nn.Conv2d(dim, dim, 1)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x_tar, x_ref):
        # 仅利用 Tar，防止 Ref 干扰
        feat = self.main_conv(x_tar)
        mask = self.gate_conv(x_tar)
        # 门控机制：过滤掉伪影/噪声
        return self.out_proj(feat * mask), []

# ---------------------------------------------------------------------------
# --- E4: Deformable Alignment Expert (隐式配准) ---
# ---------------------------------------------------------------------------
class DeformableAlignmentExpert(nn.Module):
    """
    E4: 可变形对齐专家
    优势: 解决 MRI 序列间的 Spatial Misalignment，实现隐式配准。
    """
    def __init__(self, dim):
        super().__init__()
        # Offset 预测器: 必须同时看 Tar 和 Ref 才能知道怎么对齐
        self.offset_conv = nn.Conv2d(dim * 2, 18, 3, 1, 1) 
        self.dcn = DeformConv2d(dim, dim, 3, padding=1)
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        # 零初始化 offset，初始状态为标准卷积
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x_tar, x_ref):
        # 1. 预测偏移量: 观察 Tar 和 Ref 的差异
        offset = self.offset_conv(torch.cat([x_tar, x_ref], dim=1))
        
        # 2. 对 Ref 进行物理变形 (Warping) 以对齐 Tar
        aligned_ref = self.dcn(x_ref, offset)
        
        return self.fusion(aligned_ref), []

# ---------------------------------------------------------------------------
# --- 主模块: ARFU_MoE (Soft Routing) ---
# --- (与你的旧代码签名兼容的混合版本) ---
# ---------------------------------------------------------------------------

class ARFU_MoE(nn.Module): # ARFU_MoE: Semantic-aware Soft Fusion Mixture of Experts 
    def __init__(self, dim=96, num_experts=4, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.K = 4
        
        # 预处理
        self.pre_tar = nn.Conv2d(dim, dim, 3, 1, 1)
        self.pre_ref = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # 门控网络 (Router)
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 3, 1, 1), # 输入 Tar, Ref, Diff
            nn.SiLU(),
            nn.Conv2d(dim * 2, self.K, 1)
        )
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)
        
        self.experts = nn.ModuleList([
            # E1: Global (STSS2D) - 负责长距离依赖
            STSS2D_Expert(d_model=dim, d_state=d_state, **kwargs),
            
            # E2: Local (Gradient) - 负责高频边缘修复 (Novelty up!)
            GradientModulationExpert(dim),
            
            # E3: Target (Gated) - 负责抗伪影保真
            GatedSelectionExpert(dim),
            
            # E4: Ref (Deformable) - 负责空间对齐与纹理迁移
            DeformableAlignmentExpert(dim)
        ])
        
        # 后处理融合
        self.post_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        nn.init.constant_(self.post_conv[-1].weight, 0)

    def forward(self, tar, ref):
        # 1. 解耦与预处理
        x_tar = self.pre_tar(tar)
        x_ref = self.pre_ref(ref)
        x_diff = x_tar - x_ref
        
        # 2. 路由
        # 为什么要加 x_diff？提供“差异/误差”信号。如果 x_diff 在某区域非常大（比如 Ref 有伪影，或者没对齐），路由器会倾向于激活 E1 (Global) 或 E4 (Align)。
        # 如果 x_diff 很小（两者一致），路由器可能会激活 E3 (Target Pres) 或 E2 (Local)。显式输入差异图能让门控网络更容易学到这种判别逻辑。
        gate_in = torch.cat([x_tar, x_ref, x_diff], dim=1)
        gate_logits = self.gate_net(gate_in)
        weights = F.softmax(gate_logits, dim=1).unsqueeze(2) # [B, 4, 1, H, W]
        
        # 3. 专家计算
        # E1: 内部自融合
        y1, aux1 = self.experts[0](x_tar, x_ref)
        # E2: 融合 Tar/Ref 提取边缘
        y2, _ = self.experts[1](x_tar, x_ref)
        # E3: 只看 Tar (Gated)
        y3, _ = self.experts[2](x_tar, None)
        # E4: 只看 Ref (但 Offset 需要 Tar 参考)
        y4, _ = self.experts[3](x_tar, x_ref)
        
        # 4. 聚合
        y_stack = torch.stack([y1, y2, y3, y4], dim=1)
        y_agg = (y_stack * weights).sum(dim=1)
        
        # --- 核心修复: 正确收集 aux 数据 ---
        # 列表第一个元素是本层 (ARFU) 的 logits
        all_aux_data = [gate_logits] if self.training else [weights.squeeze(2)]
        
        # 列表后续元素是内部专家 (STSS2D) 的 logits
        if self.training and aux1:
            all_aux_data.extend(aux1) # [Gate_ARFU, Gate_STSS2D]
        
        # 5. 残差输出
        return tar + self.post_conv(y_agg), all_aux_data

class CLFR(nn.Module):
    # ... (CLFR 代码保持不变) ...
    def __init__(self, dim):
        super().__init__()
    def reconstructing_procedure(self, f_h, f_m):
        _, c, h, w = f_h.shape 
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)
        f_m_T = torch.transpose(f_m, 2, 1)
        matrix_hm = torch.matmul(f_m_T, f_h)
        l2_m = torch.norm(matrix_hm)
        matrix_hm = torch.tanh(matrix_hm / l2_m)
        f_refine_h = torch.matmul(f_m, matrix_hm) + f_h
        f_refine_h_T = torch.transpose(f_refine_h, 2, 1)
        matrix_mh = torch.matmul(f_refine_h_T, f_m)
        l2_h = torch.norm(matrix_mh)
        matrix_mh = torch.tanh(matrix_mh / l2_h)
        f_refine_m = torch.matmul(f_refine_h, matrix_mh) + f_m
        return f_refine_h.view(-1, c, h, w), f_refine_m.view(-1, c, h, w)
    def forward(self, f_h, f_m):
        f_refine_h, f_refine_m = self.reconstructing_procedure(f_h, f_m)
        return f_refine_h, f_refine_m

class WavMCVM(nn.Module): # WavMCVM: Wavelet-guided Hybrid MoE Network
    def __init__(self, upscale, 
                inchans,
                outchans,
                dim,
                depth,
                d_state, 
                drop, 
                attn_drop,
                drop_path,
                norm_layer,
                patch_size,
                patch_norm,
                router_jitter_noise,
                downsample=None,
                use_checkpoint=False):
        super(WavMCVM, self).__init__()
        self.upscale = upscale
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=False)
        self.conv2d = Conv2D(in_chl=inchans, nf=dim, n_blks=depth)
        self.conv_first = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_second = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=2, padding=1)
        self.conv_third = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=2, padding=1)
        self.reduce_conv = nn.Conv2d(192, 96, kernel_size=1) 
        self.conv_last = nn.Conv2d(in_channels=dim, out_channels=outchans, kernel_size=(3, 3), stride=1, padding=1)
        
        stvm_args = {
            "inchans": dim, "outchans": dim, "dim": dim, "depth": depth, 
            "d_state": d_state, "drop": drop, "attn_drop": attn_drop, 
            "drop_path": drop_path[0], 
            "norm_layer": norm_layer, "patch_size": patch_size, 
            "patch_norm": patch_norm, "is_cross": False, 
            "router_jitter_noise": router_jitter_noise, "downsample": downsample, 
            "use_checkpoint": use_checkpoint
        }
        self.stvm_tar = STVMUnit(**stvm_args) 
        self.stvm_ref_w1 = STVMUnit(**stvm_args)
        self.stvm_ref_w2 = STVMUnit(**stvm_args)
        self.stvm_ref_r0 = STVMUnit(**stvm_args)
        self.stvm_ref_r1 = STVMUnit(**stvm_args)
        self.stvm_ref_r2 = STVMUnit(**stvm_args)

        hybrid_args = {
            "inchans": dim, "outchans": dim, "dim": dim, "depth": depth,
            "d_state": d_state, "drop": drop, "attn_drop": attn_drop,
            "drop_path": drop_path, 
            "norm_layer": norm_layer, "patch_size": patch_size,
            "patch_norm": patch_norm, "router_jitter_noise": router_jitter_noise,
            "downsample": None, "use_checkpoint": use_checkpoint
        }
        self.hybridstm_0 = HybridSTM(**hybrid_args)
        self.hybridstm_1 = HybridSTM(**hybrid_args)
        self.hybridstm_2 = HybridSTM(**hybrid_args)

        arfu_args = {"dim": dim, "d_state": d_state}
        self.arfu_0 = ARFU_MoE(**arfu_args)
        self.arfu_1 = ARFU_MoE(**arfu_args)
        self.arfu_2 = ARFU_MoE(**arfu_args)
        # self.arfu_0 = wo_DAE(**arfu_args)
        # self.arfu_1 = wo_DAE(**arfu_args)
        # self.arfu_2 = wo_DAE(**arfu_args)

    def refsobel(self, ref):
        # 假设ref是torch.Tensor
        ref = ref.detach().cpu().numpy()  # 转换为numpy数组
        sobel_channels = []
        for c in range(ref.shape[1]):  # 针对每个通道分别计算Sobel
            ref_sx = cv2.Sobel(ref[:, c, :, :], cv2.CV_32F, 1, 0)
            ref_sy = cv2.Sobel(ref[:, c, :, :], cv2.CV_32F, 0, 1)
            ref_s = cv2.addWeighted(ref_sx, 0.5, ref_sy, 0.5, 0)
            sobel_channels.append(ref_s)

        ref_sobel = np.stack(sobel_channels, axis=1)  # 合并通道
        ref_sobel = torch.from_numpy(ref_sobel).cuda()  # 转换回torch.Tensor并移动到原设备
        return ref_sobel

    def wavelet_high_freq(self, image):
        # Convert tensor to NumPy for wavelet transform
        img_np = image.detach().cpu().numpy()  # 转换为 NumPy 数组 (N, C, H, W)
        high_freq_components = []
        
        # 对每个通道进行小波变换，并提取高频特征
        for c in range(img_np.shape[1]):
            coeffs = pywt.dwt2(img_np[:, c, :, :], 'haar')  # 对每个通道进行2D小波变换
            LL, (LH, HL, HH) = coeffs  # LL为低频分量, LH/HL/HH为高频分量
            high_freq_components.append(HH)  # 提取最高频分量HH

        # 将每个通道的高频分量合并回张量
        high_freq_np = np.stack(high_freq_components, axis=1)  # (N, C, H, W)
        high_freq_tensor = torch.from_numpy(high_freq_np).cuda()  # 转换回 tensor

        return high_freq_tensor
    
    def fourier_high_freq(self, image):
        # Convert tensor to NumPy for Fourier transform
        img_np = image.detach().cpu().numpy()  # 转换为 NumPy 数组 (N, C, H, W)
        high_freq_components = []
        
        # 对每个通道进行傅里叶变换，并提取高频特征
        for c in range(img_np.shape[1]):
            fft_img = np.fft.fft2(img_np[:, c, :, :])  # 对每个通道进行2D傅里叶变换
            fft_shifted = np.fft.fftshift(fft_img)  # 将低频分量移到频谱中心

            # 创建一个掩膜，仅保留高频分量
            rows, cols = fft_shifted.shape[-2:]
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), dtype=np.float32)
            r = min(rows, cols) // 4  # 选择高频范围
            mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # 中心区域设置为0 (低频)
            
            # 应用掩膜并反傅里叶变换
            high_freq_fft = fft_shifted * mask
            high_freq_img = np.fft.ifft2(np.fft.ifftshift(high_freq_fft))  # 反变换回空间域
            high_freq_img = np.abs(high_freq_img)  # 取绝对值，去除复数部分
            high_freq_components.append(high_freq_img)  # 提取高频分量

        # 将每个通道的高频分量合并回张量
        high_freq_np = np.stack(high_freq_components, axis=1)  # (N, C, H, W)
        
        # Ensure the result is float32 to match the convolution layers
        high_freq_tensor = torch.from_numpy(high_freq_np).float().cuda()  # 转换为 float32
        
        return high_freq_tensor

    def forward(self, tar, ref):
        all_aux_data = {}
        
        tar_lr_conv = self.conv2d(tar)
        tar_lr, aux_stvm = self.stvm_tar(tar_lr_conv)
        if aux_stvm:
            all_aux_data['stvm_tar'] = aux_stvm 

        ref_wavelet_1_freq = self.wavelet_high_freq(ref)
        ref_wavelet_1_conv = self.conv_first(ref_wavelet_1_freq)
        ref_wavelet_1, aux_w1 = self.stvm_ref_w1(ref_wavelet_1_conv)
        if aux_w1:
            all_aux_data['stvm_ref_w1'] = aux_w1
            
        ref_wavelet_2_conv = self.conv_second(ref_wavelet_1)
        ref_wavelet_2, aux_w2 = self.stvm_ref_w2(ref_wavelet_2_conv)
        if aux_w2:
            all_aux_data['stvm_ref_w2'] = aux_w2

        ref_0_conv = self.conv_first(ref)
        ref_0, aux_r0 = self.stvm_ref_r0(ref_0_conv)
        if aux_r0:
            all_aux_data['stvm_ref_r0'] = aux_r0
            
        ref_1_conv = self.conv_second(ref_0)
        ref_1, aux_r1 = self.stvm_ref_r1(ref_1_conv)
        if aux_r1:
            all_aux_data['stvm_ref_r1'] = aux_r1
            
        ref_2_conv = self.conv_third(ref_1)
        ref_2, aux_r2 = self.stvm_ref_r2(ref_2_conv)
        if aux_r2:
            all_aux_data['stvm_ref_r2'] = aux_r2

        if self.upscale == 2:
            fuse_1, aux_1 = self.arfu_1(tar_lr, ref_1)
            if aux_1: all_aux_data['arfu_1'] = aux_1
                
            fuse_1, aux_h1 = self.hybridstm_1(fuse_1, style=ref_wavelet_1)
            if aux_h1: all_aux_data['hybridstm_1'] = aux_h1

            fuse_1 = self.up(fuse_1)

            fuse_2, aux_2 = self.arfu_2(fuse_1, ref_0)
            if aux_2: all_aux_data['arfu_2'] = aux_2
                
            fuse_2, aux_h2 = self.hybridstm_2(fuse_2, style=F.interpolate(ref_wavelet_1, scale_factor=2, mode='bilinear', align_corners=False))
            if aux_h2: all_aux_data['hybridstm_2'] = aux_h2

            out = self.conv_last(fuse_2)

        if self.upscale == 4:
            fuse_0, aux_0 = self.arfu_0(tar_lr, ref_2)
            if aux_0: all_aux_data['arfu_0'] = aux_0
                
            fuse_0, aux_h0 = self.hybridstm_0(fuse_0, style=ref_wavelet_2)
            if aux_h0: all_aux_data['hybridstm_0'] = aux_h0

            fuse_0 = self.up(fuse_0)

            fuse_1, aux_1 = self.arfu_1(fuse_0, ref_1)
            if aux_1: all_aux_data['arfu_1'] = aux_1
                
            fuse_1, aux_h1 = self.hybridstm_1(fuse_1, style=ref_wavelet_1)
            if aux_h1: all_aux_data['hybridstm_1'] = aux_h1
                
            fuse_1 = self.up(fuse_1)

            fuse_2, aux_2 = self.arfu_2(fuse_1, ref_0)
            if aux_2: all_aux_data['arfu_2'] = aux_2
                
            fuse_2, aux_h2 = self.hybridstm_2(fuse_2, style=F.interpolate(ref_wavelet_1, scale_factor=2, mode='bilinear', align_corners=False))
            if aux_h2: all_aux_data['hybridstm_2'] = aux_h2
            
            out = self.conv_last(fuse_2)

        return out, all_aux_data

        
if __name__ == '__main__':
    # --- 1. 设置参数 ---
    upscale = 4
    inchans = 3
    outchans = 3
    height = 128
    width = 128
    dim = 96
    depths = 2 # 注意：你的 __init__ 只用了 depth，没用 depths
    drop_rate = 0.
    drop_path = [0, 0.1] # 注意：你的 __init__ 只用了 drop_path[0]
    norm_layer = nn.LayerNorm
    attn_drop_rate = 0.
    d_state = 16
    patch_size = 4
    patch_norm = True
    router_jitter_noise = 0.01 

    # --- 2. 初始化模型和输入 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tar = torch.randn((1, 3, height, width)).to(device)
    ref = torch.randn((1, 3, height * upscale, width * upscale)).to(device)
    
    # 假设 WavMCVM, ARFU_MoE, STSS2D 已经在这个文件中定义或导入
    model = WavMCVM(upscale=upscale,
                    inchans=inchans,
                    outchans=outchans,
                    dim=dim,
                    depth=depths, # 传递 depths
                    d_state=d_state, 
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path, # 传递 drop_path 列表
                    norm_layer=norm_layer,
                    patch_size=patch_size,
                    patch_norm=patch_norm,
                    router_jitter_noise=router_jitter_noise,
                    downsample=None,
                    use_checkpoint=False,).to(device)
    
    # *** 修改 ***: 移除 set_routing_strategy。
    # 模型的行为现在由 .train() 和 .eval() 控制。
    model.eval() 

    # --- 3. 计算参数量 ---
    # print("--- 正在计算参数量 ---")
    
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_inactive_params = 0
    
    # # 遍历所有模块，查找 STSS2D 和 ARFU_MoE 实例
    # for module in model.modules():
    #     if isinstance(module, STSS2D):
    #         # STSS2D 总是 "hard" (STE)，K=4, k=2
    #         K = module.K
    #         k = module.top_k
            
    #         # 找到所有 K 个专家共享的参数
    #         all_expert_params = 0
    #         all_expert_params += module.x_proj_weight.numel()
    #         all_expert_params += module.dt_projs_weight.numel()
    #         all_expert_params += module.dt_projs_bias.numel()
    #         all_expert_params += module.A_logs.numel()
    #         all_expert_params += module.Ds.numel()
            
    #         if hasattr(module, 'style_proj_weight'):
    #             all_expert_params += module.style_proj_weight.numel()

    #         # 计算非激活专家的参数量
    #         inactive_params = all_expert_params * (K - k) / K
    #         total_inactive_params += inactive_params
        
    #     elif isinstance(module, ARFU_MoE):
    #         # *** 修改 ***
    #         # ARFU_MoE (外层) 总是 "soft" 路由。
    #         # 它的所有专家 (K=4) 都是激活的 (被加权)。
    #         # 因此，它自身没有非激活参数 (inactive_params = 0)。
    #         # 我们只计算其 STSS2D_Expert 内部的非激活参数，
    #         # 这在上面的 `isinstance(module, STSS2D)` 循环中已经自动处理了。
    #         pass

    # activated_params = total_params - total_inactive_params
    
    # print(f"总参数量 (Total Params): {total_params / 1e6:.2f} M")
    # print(f"可激活参数量 (Activated Params, STSS2D Hard Top-2): {activated_params / 1e6:.2f} M")
    # print(f"  (注意: ARFU_MoE 总是 'soft' 路由, STSS2D 总是 'hard' 路由)")


    # # --- 4. 计算 FLOPs (仅总 FLOPs) ---
    # print("\n--- 正在计算 FLOPs ---")
    
    # # 4.1. 总 FLOPs
    # # *** 修改 ***: 确保模型处于 eval 模式，
    # # 因为 train 模式返回 tuple[out, logits]，会使 FlopCountAnalysis 失败。
    # model.eval() 
        
    # flops_analyzer = FlopCountAnalysis(model, (tar, ref))
    # total_flops = flops_analyzer.total()
    # print(f"总 FLOPs (Total FLOPs): {total_flops / 1e9:.2f} G")
    
    # # 4.2. 可激活 FLOPs 计算已按要求移除

    # --- 5. 计算推理时间 ---
    print("\n--- 正在计算推理时间 ---")
    
    # *** 修改 ***: 移除 set_routing_strategy
    # 确保模型处于 .eval() 模式，这才是真实的推理场景
    model.eval()
    
    # 预热
    warmup_iterations = 10
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(tar, ref) 
            
    torch.cuda.synchronize() # 等待 GPU 完成
    
    # 计时
    num_iterations = 50
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(tar, ref) 
            
    torch.cuda.synchronize() # 确保所有 GPU 操作完成
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
    # *** 修改 ***: 更新描述
    print(f"推理时间 (Inference Time, Eval Mode): {avg_time_ms:.2f} ms")
