import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
# from model.vmamba_moe import *
from vmamba_moe import *

from timm.models.layers import DropPath, trunc_normal_

        
class STVMUnit(nn.Module):
    def __init__(self, 
                inchans,
                outchans,
                dim,
                depth,
                d_state, # 20240109
                drop, 
                attn_drop,
                drop_path,
                norm_layer,
                patch_size,
                patch_norm,
                is_cross,
                router_jitter_noise,
                downsample=None,
                use_checkpoint=False):
        super(STVMUnit, self).__init__()
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=inchans, embed_dim=dim,
            norm_layer=norm_layer if patch_norm else None)
        self.Layer = STVSSLayer(
                dim=dim,
                depth=depth,
                d_state=d_state, # 20240109
                drop=drop, 
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                is_cross=is_cross,
                router_jitter_noise=router_jitter_noise,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
        self.pos_drop = nn.Dropout(p=drop)
        self.final_conv = nn.Conv2d(dim, outchans, kernel_size=1)
        
        # --- 关键: 零初始化最后的输出层 ---
        # (这有助于残差学习)
        nn.init.constant_(self.final_conv.weight, 0)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x, style=None):
        x = self.patch_embed(x)
        if style is not None:
            style = self.patch_embed(style)
        
        x, aux_data = self.Layer(x, style=style)
            
        x = self.pos_drop(x)   
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)

        # 始终返回 (output, aux_data)
        return x, aux_data

class HybridSTM(nn.Module): # Hybrid CNN-VMamba Block
    def __init__(self, inchans,
                outchans,
                dim,
                depth,
                d_state, # 20240109
                drop, 
                attn_drop,
                drop_path,
                norm_layer,
                patch_size,
                patch_norm,
                router_jitter_noise,
                downsample=None,
                use_checkpoint=False,
                **kwargs): # 确保 **kwargs 被接收
        super(HybridSTM, self).__init__()
        self.stvmunit1 = STVMUnit(inchans=inchans, outchans=outchans, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path[0], norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, is_cross=True, router_jitter_noise=router_jitter_noise, downsample=downsample, use_checkpoint=use_checkpoint, **kwargs)
        self.stvmunit2 = STVMUnit(inchans=dim, outchans=dim, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path[1], norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, is_cross=True, router_jitter_noise=router_jitter_noise, downsample=downsample, use_checkpoint=use_checkpoint, **kwargs)

    def forward(self, x, style=None):
        
        all_aux_data = []
        residual = x
        
        # --- 正确的修改 ---
        # STVMUnit 总是返回 (output, aux_data)            
        stvmunit_feat, aux_list_1 = self.stvmunit1(x, style=style)
        if aux_list_1: # aux_list_1 是一个列表
            all_aux_data.extend(aux_list_1)
            
        stvmunit_feat_2, aux_list_2 = self.stvmunit2(stvmunit_feat, style=style)
        if aux_list_2: # aux_list_2 是一个列表
            all_aux_data.extend(aux_list_2)
            
        x = stvmunit_feat_2 + residual
        # --- 修改结束 ---

        # 始终返回 (output, aux_data_list)
        return x, all_aux_data
