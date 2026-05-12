import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets.dataset import BraTs_datasets
import tensorboardX
from tensorboardX import SummaryWriter

from engine import *
import sys

from utils import *
from configs.config_setting import setting_config
from setting import initialize_distributed, partition_dataset

import warnings
warnings.filterwarnings("ignore")

class NoOpLogger:
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            pass
        return no_op

class NoOpWriter:
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            pass
        return no_op


def main(config):

    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')


    global logger, writer
    if getattr(config, 'enable_logging', True):  # 默认为开启日志
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists(outputs):
            os.makedirs(outputs, exist_ok=True)
        logger = get_logger('train', log_dir)
        
        # 启动SummaryWriter
        if not dist.is_initialized() or dist.get_rank() == 0:
            summary_dir = os.path.join(config.work_dir, 'summary')
            os.makedirs(summary_dir, exist_ok=True)
            writer = SummaryWriter(summary_dir)
        else:
            writer = NoOpWriter() 

        log_config_info(config, logger)
    else:
        logger = NoOpLogger()
        writer = NoOpWriter()



    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------GPU init----------#')
    local_rank, world_size, device = initialize_distributed(config.num_gpus)
    print(f"Using {device} of {world_size}")
    set_seed(config.seed)
    torch.cuda.empty_cache()

    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Preparing dataset----------#')    
    train_dataset = BraTs_datasets(config.data_path, config, train=True)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size, 
                                    pin_memory=True,
                                    num_workers=config.num_workers, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=config.num_workers)
    val_dataset = BraTs_datasets(config.data_path, config, train=False)
    if dist.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                pin_memory=True, 
                                num_workers=config.num_workers, sampler=val_sampler)
    else:
        val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)


    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'gdcsr':
        from model.gated_delta_sr import DualInputConvNeXtUNet as GDCSR
        model = GDCSR(
            in_ch=model_cfg['inchans'],
            out_ch=model_cfg['outchans'],
            scale=model_cfg['upscale'],
            dims=(48,72,96,144),
            depths=(2, 2, 2, 2),
            mlp_ratio=2.0,
        )
    elif config.network == 'wavmcvm_moe':
        from model.multisup_moe import WavMCVM as WavMCVM_MoE
        model = WavMCVM_MoE(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
            router_jitter_noise=model_cfg['router_jitter_noise'],
        )
    elif config.network == 'wavmcvm':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_gmp':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_woarfu':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_wostyle':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_ref':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_ref_mean':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_ref_std':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_sobel':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'wavmcvm_fourier':
        from model.multisup import WavMCVM
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'minet':
        from model.MINet import MINet
        model = MINet(scale=model_cfg['upscale'], n_resgroups=2,n_resblocks=2, n_feats=64)
    elif config.network == 'mcmrsr':
        from model.mcmrsr import McMRSR
        model = McMRSR(upscale=model_cfg['upscale'], img_size=(112, 112),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    elif config.network == 'swinir':
        from model.swinir import SwinIR
        model = SwinIR(upscale=model_cfg['upscale'], in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    elif config.network == 'wavtrans':
        from model.wavtrans import WavTrans
        model = WavTrans(upscale=model_cfg['upscale'], img_size=(56, 56),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    elif config.network == 'dcamsr':
        from model.dcamsr import DCAMSR
        model = DCAMSR(scale=model_cfg['upscale'])
    elif config.network == 'ecfnet':
        from model.ecfnet import ECFNet
        model = ECFNet(scale=model_cfg['upscale'])
    elif config.network == 'a2_cdic':
        from model.CDic_Align_l import CDic_Align
        model = CDic_Align(scale=model_cfg['upscale'])
    elif config.network == 'srcnn':
        from model.srcnn import SRCNN
        model = SRCNN(upscale=model_cfg['upscale'])
    elif config.network == 'edt':
        from model.edt import EDT
        model = EDT(upscale=model_cfg['upscale'])
    elif config.network == 'hat':
        from model.hat import HAT
        model = HAT(img_size=64, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upscale=model_cfg['upscale'])
    elif config.network == 'rdn':
        from model.rdn import RDN
        model = RDN(scale_factor=model_cfg['upscale'], num_channels=3, num_features=64, growth_rate=64, num_blocks=8, num_layers=4)
    elif config.network == 'edsr':
        from model.edsr import EDSR
        model = EDSR(scale_factor=model_cfg['upscale'])
    else: raise Exception('network in not right!')
    model = model.to(device)

    if dist.is_initialized():
        model = DistributedDataParallel(model, find_unused_parameters=True)


    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_psnr = float('-inf')


    # 断点续训
    if os.path.exists(resume_model) and config.resume_training == True:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        if dist.is_initialized():
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.module.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.module.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, max_psnr, min_epoch, loss = checkpoint['min_loss'], checkpoint['max_psnr'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, max_psnr: {max_psnr:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        print('#----------Training----------#')
        
    # 定义中间保存间隔
    save_interval = 50
    
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        # is_milestone = (epoch % save_interval == 0)
        is_milestone = False
        
        loss_torch, avg_psnr = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                device,
                save_intermediate=is_milestone
            )

        loss_torch = loss_torch.tolist()

        if getattr(config, 'enable_logging', True):
            if dist.is_initialized():
                if torch.distributed.get_rank() == 0:
                    loss_torch_epoch = loss_torch[0] / loss_torch[1]
                    logger.info(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
                    print(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
                    
                    # START: 添加 TensorBoard 写入代码 (针对分布式 Rank 0)
                    writer.add_scalar('Loss/Train_Validation', loss_torch_epoch, epoch)
                    writer.add_scalar('Metric/Validation_PSNR', avg_psnr, epoch)
                    
                    if is_milestone:
                        milestone_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                        torch.save(model.module.state_dict(), milestone_path)
                        logger.info(f"Saved intermediate checkpoint to {milestone_path}")
                        print(f"Saved intermediate checkpoint to {milestone_path}")

                    if avg_psnr > max_psnr:
                        torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                        max_psnr = avg_psnr
                        min_epoch = epoch
                        min_loss = loss_torch_epoch

                    torch.save(
                        {
                            'epoch': epoch,
                            'min_loss': min_loss,
                            'max_psnr': max_psnr,
                            'min_epoch': min_epoch,
                            'loss': loss_torch_epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(checkpoint_dir, 'latest.pth')) 
            else:
                loss_torch_epoch = loss_torch[0] / loss_torch[1]
                logger.info(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
                print(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
                
                # START: 添加 TensorBoard 写入代码 (针对非分布式模式)
                writer.add_scalar('Loss/Train_Validation', loss_torch_epoch, epoch)
                writer.add_scalar('Metric/Validation_PSNR', avg_psnr, epoch)
                
                if is_milestone:
                    milestone_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                    torch.save(model.module.state_dict(), milestone_path)
                    logger.info(f"Saved intermediate checkpoint to {milestone_path}")
                    print(f"Saved intermediate checkpoint to {milestone_path}")

                if avg_psnr > max_psnr:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                    max_psnr = avg_psnr
                    min_epoch = epoch
                    min_loss = loss_torch_epoch

                torch.save(
                        {
                            'epoch': epoch,
                            'min_loss': min_loss,
                            'max_psnr': max_psnr,
                            'min_epoch': min_epoch,
                            'loss': loss_torch_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(checkpoint_dir, 'latest.pth')) 
        else:
            pass

    # ----------Testing----------
    if dist.is_initialized():
        torch.distributed.barrier()  # 等待所有rank完成训练与保存
        best_model_path = os.path.join(checkpoint_dir, 'best.pth')

        if os.path.exists(best_model_path):
            if torch.distributed.get_rank() == 0:
                print('#----------Testing----------#')

            # 所有rank都加载相同模型权重
            best_weight = torch.load(best_model_path, map_location='cpu')
            model.module.load_state_dict(best_weight)

            # 每个rank都执行测试（并行推理）
            loss_torch, avg_psnr = test_one_epoch(
                val_loader, model, criterion, logger, config, device
            )

            # 只在rank 0上打印日志与重命名文件
            if torch.distributed.get_rank() == 0:
                new_name = os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
                os.rename(best_model_path, new_name)
                print(f'[Info] Renamed best model to {new_name}')
                print(f'[Test] Average Loss: {loss_torch:.4f}, PSNR: {avg_psnr:.4f}')
        else:
            if torch.distributed.get_rank() == 0:
                print(f'[Warning] best.pth not found at {best_model_path}, skip testing.')

        torch.distributed.barrier()  # 所有rank等待测试完成

    else:
        # 单机测试模式
        best_model_path = os.path.join(checkpoint_dir, 'best.pth')
        if os.path.exists(best_model_path):
            print('#----------Testing----------#')
            best_weight = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(best_weight)
            loss_torch, avg_psnr = test_one_epoch(
                val_loader, model, criterion, logger, config, device
            )
            new_name = os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            os.rename(best_model_path, new_name)
            print(f'[Info] Renamed best model to {new_name}')
            print(f'[Test] Average Loss: {loss_torch:.4f}, PSNR: {avg_psnr:.4f}')
        else:
            print(f'[Warning] best.pth not found, skip testing.')

    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == '__main__':
    config = setting_config
    main(config)
