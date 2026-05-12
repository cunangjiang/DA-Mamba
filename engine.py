import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from utils import calc_psnr, convert_rgb_to_y, denormalize
# from utils import save_imgs
from torchvision import transforms
import os

def compute_load_balancing_loss(gate_logits_list, K=4, alpha=1e-2):
    """
    计算负载均衡损失 (Load Balancing Loss)。
    惩罚 STSS2D (Hard MoE) 和 ARFU_MoE (Soft MoE) 的专家负载不均衡。

    此实现基于 "V-MoE" (Riquelme et al., 2021) 论文中的
    L2 负载均衡损失 (L_aux = alpha * K * sum(f_i^2))，
    它在 Soft MoE 和基于 STE 的 Hard MoE 上均有效且稳定。

    Args:
        gate_logits_list (list): 包含所有 MoE 层 logits 的列表。
                                 每个张量形状为 (B, K)。
        K (int): 专家数量 (我们的设计中 STSS2D 和 ARFU 都是 4)。
        alpha (float): 辅助损失的缩放系数。
    """
    if not gate_logits_list or alpha == 0:
        # (修复: 提供了更安全的 device-agnostic 默认值)
        return torch.tensor(0.0, dtype=torch.float32)

    # (修复: 确保 total_loss 是一个张量，以便进行设备间操作)
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    num_experts = K

    # --- 递归辅助函数 ---
    def process_item(item):
        """ 递归处理列表或张量 """
        nonlocal total_loss # 允许修改外部的 total_loss
        
        if isinstance(item, torch.Tensor):
            # --- 基础情况: 这是一个 `logits` 张量 ---
            logits = item
            
            # (修复: 确保 total_loss 和 logits 在同一设备上)
            if total_loss.device != logits.device:
                total_loss = total_loss.to(logits.device)
                
            # --- (新) 动态处理 Per-Token (4D) 和 Per-Image (2D) Logits ---
            if logits.dim() == 4:
                # 逐令牌 (Per-Token): 形状 [B, K, H, W]
                B, K_experts, H, W = logits.shape
                softmax_dim = 1       # 沿 K 维度 (dim=1) Softmax
                mean_dims = (0, 2, 3) # 沿 B, H, W 维度求平均
            elif logits.dim() == 2:
                # 逐图像 (Per-Image): 形状 [B, K]
                B, K_experts = logits.shape
                softmax_dim = -1      # 沿 K 维度 (dim=-1) Softmax
                mean_dims = 0         # 沿 B 维度求平均
            else:
                print(f"警告: MoE logits 形状 {logits.shape} 不支持, 跳过。")
                return # 跳过这个张量
            # --- (新) 修改结束 ---

            if K_experts != num_experts:
                print(f"警告: MoE 专家数量不匹配 (K={num_experts} vs {K_experts})，跳过此层。")
                return # 跳过这个张量
            
            # (新) 使用动态维度计算 f_i
            f_i = F.softmax(logits, dim=softmax_dim).mean(dim=mean_dims) # (K)
            loss_aux_layer = (f_i * f_i).sum() * num_experts
            
            total_loss = total_loss + loss_aux_layer # 累加损失
        
        elif isinstance(item, list):
            # --- 递归情况: 这是一个嵌套列表 ---
            for sub_item in item:
                process_item(sub_item) # 递归调用
                
        elif isinstance(item, dict):
                # --- 新增: 递归情况: 这是一个字典 ---
                for sub_item in item.values():
                    process_item(sub_item) # 递归调用
        
        elif item is None:
            # 忽略 None (例如在 eval 模式下的 STSS2D_Expert)
            pass
        
        else:
            print(f"警告: compute_load_balancing_loss 遇到未知类型: {type(item)}，已忽略。")
    # --- 启动递归处理 ---
    process_item(gate_logits_list)

    return total_loss * alpha

# ===== [MOD-2] =====
def gradient_loss(pred, target):
    """
    一阶差分梯度损失：
    同时约束 x / y 两个方向的梯度一致性。

    pred, target: [B, C, H, W]
    return: scalar
    """
    # x 方向梯度（宽度方向）
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # y 方向梯度（高度方向）
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)

    return loss_dx + loss_dy

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer,
                    device):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        ori, ref, ref_lr, tar, img_name = data
        ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()

        # Forward pass (WavMCVM 在 train() 模式下返回 (output, all_aux_logits))
        if config.network == 'wavmcvm_moe':
            sr_output, all_aux_logits = model(tar, ref)
        else:
            sr_output = model(tar, ref)
        # sr_output = model(tar)
        # sr_output = model(tar, ref_lr, ref)
        # sr_output, _, _, _, _, _, _, _ = model(tar, ref)

        
        # Compute losses for segmentation and super-resolution (or other tasks if needed)
        loss_main = criterion(sr_output, ori)
        
        # gradient loss
        if getattr(config, 'use_gradient_loss', False):
            loss_grad = gradient_loss(sr_output, ori)
            loss_recon = loss_main + config.gradient_loss_weight * loss_grad
        else:
            loss_grad = torch.tensor(0.0, device=sr_output.device)
            loss_recon = loss_main
        
        # --- 新增: 计算负载均衡损失 ---
        if config.network == 'wavmcvm_moe':
            loss_aux = compute_load_balancing_loss(
            all_aux_logits, 
            K=4, # STSS2D 和 ARFU_MoE 都是 K=4
            alpha=config.moe_aux_loss_alpha # 从 config 读取 alpha
        )
            loss = loss_main + loss_aux
        else:
            loss = loss_main
            
        loss.backward()
        optimizer.step()
        
        if torch.distributed.get_rank() == 0:
            writer.add_scalar('Loss/Train_step', loss.item(), step)
            writer.add_scalar('Loss/Train_Main', loss_main.item(), step)
            
            if getattr(config, 'use_gradient_loss', False):
                writer.add_scalar(
                    'Loss/Train_Grad', 
                    loss_grad.item(), 
                    step
                )
                
                writer.add_scalar(
                    'Loss/Train_Recon_Total',
                    loss_recon.item(),
                    step
                )
            
            # ===== wavmcvm_moe =====
            if config.network == 'wavmcvm_moe':
                writer.add_scalar('Loss/Train_Aux_MoE', loss_aux.item(), step)

                if all_aux_logits: 
                    try:
                        # --- 1. 监控 STSS2D (Hard MoE) ---
                        # 检查是否存在且列表不为空
                        if 'stvm_tar' in all_aux_logits and len(all_aux_logits['stvm_tar']) > 0:
                            writer.add_histogram('Logits_STSS2D/tar_input', all_aux_logits['stvm_tar'][0], step)

                        if 'stvm_ref_r0' in all_aux_logits and len(all_aux_logits['stvm_ref_r0']) > 0:
                            writer.add_histogram('Logits_STSS2D/ref_input_r0', all_aux_logits['stvm_ref_r0'][0], step)
                        
                        if 'hybridstm_0' in all_aux_logits and len(all_aux_logits['hybridstm_0']) > 0:
                            writer.add_histogram('Logits_STSS2D/hybrid_stm_0', all_aux_logits['hybridstm_0'][0], step)
                            
                        if 'hybridstm_1' in all_aux_logits and len(all_aux_logits['hybridstm_1']) > 0:
                            writer.add_histogram('Logits_STSS2D/hybrid_stm_1', all_aux_logits['hybridstm_1'][0], step)
                            
                        if 'hybridstm_2' in all_aux_logits and len(all_aux_logits['hybridstm_2']) > 0:
                            writer.add_histogram('Logits_STSS2D/hybrid_stm_2', all_aux_logits['hybridstm_2'][0], step)

                        # --- 2. 监控 ARFU_MoE (Soft MoE) ---
                        
                        # ARFU_0
                        if 'arfu_0' in all_aux_logits and len(all_aux_logits['arfu_0']) >= 1:
                            # 第 0 个元素是 ARFU 自己的 gate
                            writer.add_histogram('Logits_ARFU/fuse_0_deepest', all_aux_logits['arfu_0'][0], step)
                            # 第 1 个元素是 E1 (STSS2D) 的 gate (如果存在)
                            if len(all_aux_logits['arfu_0']) >= 2:
                                writer.add_histogram('Logits_STSS2D_in_ARFU/arfu_0_expert', all_aux_logits['arfu_0'][1], step)
                        
                        # ARFU_2
                        if 'arfu_2' in all_aux_logits and len(all_aux_logits['arfu_2']) >= 1:
                            writer.add_histogram('Logits_ARFU/fuse_2_final', all_aux_logits['arfu_2'][0], step)
                            if len(all_aux_logits['arfu_2']) >= 2:
                                writer.add_histogram('Logits_STSS2D_in_ARFU/arfu_2_expert', all_aux_logits['arfu_2'][1], step)
                        
                    except Exception as e:
                        if step < config.print_interval * 2:
                            print(f"[TensorBoard] 警告: 无法解析 MoE logits 字典。错误: {e}")
                            
            # ===== gdcsr: 记录 DGCF alpha 参数 =====
            if config.network == 'gdcsr':
                try:
                    # 记录 skip-level DGCF 的 alpha
                    dgcf_blocks = {
                        'skip3': model.module.decoder.skip_fuse3.dgcf if hasattr(model, 'module') else model.decoder.skip_fuse3.dgcf,
                        'skip2': model.module.decoder.skip_fuse2.dgcf if hasattr(model, 'module') else model.decoder.skip_fuse2.dgcf,
                        'skip1': model.module.decoder.skip_fuse1.dgcf if hasattr(model, 'module') else model.decoder.skip_fuse1.dgcf,
                    }

                    for name, dgcf in dgcf_blocks.items():
                        writer.add_scalar(f'DGCFAlpha/{name}/alpha_out', dgcf.alpha_out.item(), step)

                except Exception as e:
                    if step < config.print_interval * 2:
                        print(f"[TensorBoard] 警告: 无法记录 gdcsr 的 DGCF alpha。错误: {e}")
        
        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if iter % config.print_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            log_info = f'train: epoch {epoch}, iter:{iter}, step:{step}, loss:{loss.item():.4f}, lr:{now_lr}'
            print(log_info)
            logger.info(log_info)

    scheduler.step() 
    current_lr = optimizer.param_groups[0]['lr']
    if torch.distributed.get_rank() == 0:
        writer.add_scalar('Meta/LearningRate', current_lr, step)
        
    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    device,
                    save_intermediate=False):
    # switch to evaluate mode
    model.eval()
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    psnr_values = []
    
    if save_intermediate and (not dist.is_initialized() or dist.get_rank() == 0):
        intermediate_dir = os.path.join(config.work_dir, 'intermediate_results', f'epoch_{epoch}')
        os.makedirs(intermediate_dir, exist_ok=True)
        
    with torch.no_grad():
        for data in tqdm(test_loader):
            ori, ref, ref_lr, tar, img_name = data
            ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()
            
            # Forward pass
            if config.network == 'wavmcvm_moe':
                sr_output, _ = model(tar, ref)
            else:
                sr_output = model(tar, ref)
            # sr_output = model(tar)
            # sr_output = model(tar, ref_lr, ref)
            # sr_output, _, _, _, _, _, _, _ = model(tar, ref)

            # Compute loss
            loss = criterion(sr_output, ori)
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

            sr_output_y = convert_rgb_to_y(denormalize(sr_output.squeeze(0)), dim_order='chw')
            ori = convert_rgb_to_y(denormalize(ori.squeeze(0)), dim_order='chw')

            psnr_val = calc_psnr(ori, sr_output_y)
            psnr_values.append(psnr_val)
            
            if save_intermediate: 
                if not dist.is_initialized() or dist.get_rank() == 0:
                    # 保存 SR 图像
                    sr_img = sr_output.squeeze(0).cpu()
                    sr_img = torch.clamp(sr_img, min=-1, max=1)
                    sr_img = (sr_img + 1) / 2.0
                    sr_img_pil = transforms.ToPILImage()(sr_img)
                    sr_img_pil.save(os.path.join(intermediate_dir, img_name[0]))

   
    if dist.is_initialized():
        psnr_tensor = torch.tensor(psnr_values, dtype=torch.float32, device=device)
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
        num_samples = len(test_loader) * config.num_gpus
        avg_psnr = psnr_tensor.sum().item() / num_samples
    else:
        avg_psnr = np.mean(np.array([x.cpu().item() for x in psnr_values]))

    if dist.is_initialized():
        if torch.distributed.get_rank() == 0:
            if epoch % config.val_interval == 0:
                log_info = f'val epoch: {epoch}, loss: {loss.item():.4f}, psnr: {avg_psnr:.4f}'
                print(log_info)
                logger.info(log_info)
            else:
                log_info = f'val epoch: {epoch}, loss: {loss.item():.4f}'
                print(log_info)
                logger.info(log_info)
    else:
        if epoch % config.val_interval == 0:
            log_info = f'val epoch: {epoch}, loss: {loss.item():.4f}, psnr: {avg_psnr:.4f}'
            print(log_info)
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, loss: {loss.item():.4f}'
            print(log_info)
            logger.info(log_info)
    
    return loss_torch, avg_psnr

def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    device,
                    test_data_name=None,):
    # switch to evaluate mode
    model.eval()
    val_psnr = 0
    best_score = float('-inf')
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    psnr_sum = torch.zeros(1, dtype=torch.float, device=device)
    count = torch.zeros(1, dtype=torch.float, device=device)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        iterator = tqdm(test_loader)
    else:
        iterator = test_loader
    
    with torch.no_grad():
        for i, data in enumerate(iterator):
            ori, ref, ref_lr, tar, img_name = data
            ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()

            # Forward pass
            if config.network == 'wavmcvm_moe':
                sr_output, _ = model(tar, ref)
            else:
                sr_output = model(tar, ref)
            # sr_output = model(tar)
            # sr_output = model(tar, ref_lr, ref)
            # sr_output, _, _, _, _, _, _, _ = model(tar, ref)

            # Compute losses for segmentation and super-resolution
            loss = criterion(sr_output, ori)
            
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

            sr_output_y = convert_rgb_to_y(denormalize(sr_output.squeeze(0)), dim_order='chw')
            ori = convert_rgb_to_y(denormalize(ori.squeeze(0)), dim_order='chw')

            psnr_val = calc_psnr(ori, sr_output_y)
            psnr_sum += psnr_val
            count += 1
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                sr_img = sr_output.squeeze(0).cpu()
                sr_img = torch.clamp(sr_img, min=-1, max=1)
                sr_img = (sr_img + 1) / 2.0
                sr_img = transforms.ToPILImage()(sr_img)
                sr_img.save(os.path.join(config.work_dir, 'outputs', img_name[0]))


        if dist.is_initialized():
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(psnr_sum, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            
        avg_loss = (loss_torch[0] / loss_torch[1]).item() if loss_torch[1] > 0 else 0.0
        avg_psnr = (psnr_sum / count).item() if count > 0 else 0.0

        if (not dist.is_initialized()) or (torch.distributed.get_rank() == 0):
            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)

            log_info = f'test of best model, loss: {avg_loss:.4f}, psnr: {avg_psnr:.4f}'
            print(log_info)
            logger.info(log_info)

    return (loss_torch[0] / loss_torch[1]).item(), avg_psnr
