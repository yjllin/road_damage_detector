#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - 改进的YOLOv8s模型
"""

import os
import sys
import time
import torch
import yaml
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import MultiStepLR
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.dataset import YOLODataset
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
from ultralytics.utils.loss import E2EDetectLoss
import torch.nn.functional as F
import torchvision.ops
import torchvision
import math
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.metrics import ConfusionMatrix 
from ultralytics.utils.ops import non_max_suppression
from types import SimpleNamespace # <--- 添加导入

# 从相对路径导入
from model import ImprovedYoloV8s
# from utils.datasets import create_dataloader
# from utils.general import init_seeds, increment_path
# from utils.metrics import compute_metrics

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 创建处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 配置根日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

def add_file_handler(save_dir):
    """为日志添加文件处理器"""
    log_file = Path(save_dir) / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_file

# 检查torchvision版本是否支持complete_box_iou_loss
import torchvision
HAS_COMPLETE_IOU = hasattr(torchvision.ops, 'complete_box_iou_loss')

def create_dataloaders(data_path, img_size=640, batch_size=4, workers=4):
    """
    创建训练和验证数据加载器
    
    参数:
        data_path: 数据集路径
        img_size: 图像尺寸
        batch_size: 批次大小
        workers: 数据加载的工作线程数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 类别数量
    """
    try:
        # 加载配置
        data_yaml = data_path + '/data.yaml'
        
        # 读取配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)
        data_dict.setdefault('channels', 3)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        
        # 创建数据集
        train_path = data_dict['path'] + data_dict['train_images']
        val_path = data_dict['path'] + data_dict['val_images']
        
        train_ds = YOLODataset(img_path=train_path, data=data_dict, imgsz=img_size, augment=True, cache=False)
        val_ds = YOLODataset(img_path=val_path, data=data_dict, imgsz=img_size, augment=False, cache=False)
        
        # 构建数据加载器，使用位置参数
        train_loader = build_dataloader(train_ds, batch_size, shuffle=True, workers=workers)
        val_loader = build_dataloader(val_ds, batch_size, shuffle=False, workers=workers)
        
        num_classes = data_dict['nc']
        
        logger.info(f"数据加载器创建成功:")
        logger.info(f"- 训练集: {data_dict['train_images']}")
        logger.info(f"- 验证集: {data_dict['val_images']}")
        logger.info(f"- 类别数: {num_classes}")
        logger.info(f"- 图像尺寸: {img_size}")
        logger.info(f"- 批次大小: {batch_size}")
        
        return train_loader, val_loader, num_classes
        
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        logger.error(f"数据路径: {data_path}")
        logger.error(f"配置文件: {data_yaml if 'data_yaml' in locals() else 'Not created'}")
        raise

def parse_args():
    """解析命令行参数，支持覆盖配置文件中的值"""
    parser = argparse.ArgumentParser(description='训练改进的YOLOv8s模型')
    parser.add_argument('--config', type=str, default='improved_yolov8/config.yaml', help='配置文件路径')
    parser.add_argument('--override', nargs='*', help='覆盖配置文件中的参数，格式: param=value')
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理override参数
    if args.override:
        for override in args.override:
            try:
                key, value = override.split('=')
                # 尝试转换为适当的类型
                try:
                    value = eval(value)  # 尝试转换为数字或布尔值
                except:
                    pass  # 保持为字符串
                
                # 支持嵌套键，如 'training.lr0'
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    target = target.setdefault(k, {})
                target[keys[-1]] = value
                logger.info(f"覆盖配置: {key} = {value}")
            except Exception as e:
                logger.warning(f"无法处理覆盖参数 {override}: {e}")
    
    return config

def freeze_backbone(model, freeze=True):
    """冻结或解冻backbone
    
    参数:
        model: 模型实例
        freeze: 如果为True，冻结backbone；如果为False，解冻backbone
    """
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = not freeze
    
    # 打印状态
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {frozen_params}/{total_params} 已冻结 ({frozen_params/total_params:.1%})")
    
    return model

def train(config):
    """训练主函数
    
    参数:
        config: 完整的配置字典
    """
    # 从配置中提取训练参数
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # 设置设备
    device = torch.device(f"cuda:{training_config.get('device', '0')}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 为A100/H100设置内存分配比例
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        if 'A100' in gpu_name or 'H100' in gpu_name or training_config.get('memory_fraction', 0) > 0:
            # 为高性能GPU设置更大的内存比例
            torch.cuda.set_per_process_memory_fraction(training_config.get('memory_fraction', 0.8), device)
            logger.info(f"设置GPU内存比例为{training_config.get('memory_fraction', 0.8)*100:.0f}%")
    
    # 创建输出目录
    save_dir = Path(training_config.get('output', 'runs/train')) / training_config.get('name', 'exp')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整配置
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # 创建数据加载器
    train_loader, val_loader, num_classes = create_dataloaders(
        data_path=data_config.get('path', './database/China_MotorBike'),
        img_size=model_config.get('input_size', 640),
        batch_size=training_config.get('batch_size', 16),
        workers=training_config.get('workers', 8)
    )
    
    # 初始化模型
    logger.info(f"初始化模型: 类别数={num_classes}")
    model = ImprovedYoloV8s()
    
    # 读取原始模型配置
    if hasattr(model, 'detect'):
        logger.info(f"原始模型检测头: nc={model.detect.nc}, reg_max={model.detect.reg_max}, no={model.detect.no}")
    
    # 更新模型的类别数
    if hasattr(model, 'detect'):
        logger.info(f"更新模型检测头的类别数: {num_classes}")
        if model.detect.nc != num_classes:
            orig_nc = model.detect.nc
            model.detect.nc = num_classes
            model.detect.no = num_classes + 4 * model.detect.reg_max
            logger.info(f"已更新检测头参数: nc={model.detect.nc} (原{orig_nc}), reg_max={model.detect.reg_max}, no={model.detect.no}")
    
    # 读取模型的当前配置
    if hasattr(model, 'args'):
        current_args = model.args.__dict__ if hasattr(model.args, '__dict__') else model.args
        logger.info(f"模型当前参数: {current_args}")
    
    # 更新模型参数，使用update而不是覆盖
    if hasattr(model, 'args'):
        if isinstance(model.args, dict):
            # 确保更新前保存reg_max
            reg_max_original = model.detect.reg_max if hasattr(model, 'detect') else model.args.get('reg_max', 16)
            
            model.args.update(model_config)
            
            # 确保reg_max没有被错误修改
            if 'reg_max' in model.args and hasattr(model, 'detect') and model.args['reg_max'] != model.detect.reg_max:
                logger.warning(f"reg_max不一致: args={model.args['reg_max']}, detect={model.detect.reg_max}, 以detect为准")
                model.args['reg_max'] = model.detect.reg_max
        else:
            # 确保更新前保存reg_max
            reg_max_original = model.detect.reg_max if hasattr(model, 'detect') else getattr(model.args, 'reg_max', 16)
            
            vars(model.args).update(model_config)
            
            # 确保reg_max没有被错误修改
            if hasattr(model.args, 'reg_max') and hasattr(model, 'detect') and model.args.reg_max != model.detect.reg_max:
                logger.warning(f"reg_max不一致: args={model.args.reg_max}, detect={model.detect.reg_max}, 以detect为准")
                model.args.reg_max = model.detect.reg_max
    
    # 重新生成 model.hyp 以确保它反映了所有更新，并且是一个 SimpleNamespace
    if hasattr(model, 'args'):
        if isinstance(model.args, dict):
            model.hyp = SimpleNamespace(**model.args)
        else:
            model.hyp = SimpleNamespace(**vars(model.args))
        logger.info("已从合并的 model.args 更新 model.hyp")
        logger.info(f"更新后的模型参数: {model.hyp.__dict__ if hasattr(model.hyp, '__dict__') else model.hyp}")
    
    # 检查reg_max是否正确设置，以及检测头输出维度是否正确
    if hasattr(model, 'detect'):
        expected_no = model.detect.nc + 4 * model.detect.reg_max
        if model.detect.no != expected_no:
            logger.error(f"检测头输出维度不匹配: no={model.detect.no}, 预期值={expected_no} (nc={model.detect.nc} + 4*reg_max={4*model.detect.reg_max})")
            model.detect.no = expected_no
            logger.info(f"已修复: 设置model.detect.no={model.detect.no}")
    
    # 检查reg_max是否正确设置
    if hasattr(model, 'detect') and hasattr(model, 'hyp') and hasattr(model.hyp, 'reg_max'):
        if model.detect.reg_max != model.hyp.reg_max:
            logger.warning(f"reg_max不一致: model.detect.reg_max={model.detect.reg_max}, model.hyp.reg_max={model.hyp.reg_max}")
            # 保持一致
            model.hyp.reg_max = model.detect.reg_max
            logger.info(f"已修复: 设置model.hyp.reg_max={model.hyp.reg_max}")

    # 转移模型到设备
    model = model.to(device)
    
    # 使用新的compile_model方法编译模型
    if training_config.get('use_compile', False):
        model = model.compile_model()
    
    # 冻结backbone加速训练
    if training_config.get('freeze_backbone', False):
        model = freeze_backbone(model, freeze=True)
        logger.info(f"已冻结backbone，将在第{training_config.get('unfreeze_epoch', 1)}轮解冻")

    # 确保检测头的device设置正确
    model.model[-1].device = device
    
    # 初始化损失函数
    try:
        # 首先确保模型的检测头配置正确
        if hasattr(model, 'detect'):
            model_reg_max = model.detect.reg_max
            logger.info(f"损失函数初始化前确认: model.detect.reg_max={model_reg_max}")
        else:
            model_reg_max = 16
            logger.warning(f"model没有detect属性，使用默认reg_max={model_reg_max}")
        
        criterion = DetectionLoss(model)
        logger.info("使用Ultralytics v8DetectionLoss")
        # 验证DyHead.no与Loss.no一致性
        assert model.detect.no == model.detect.nc + 4 * model.detect.reg_max, \
            "no mismatch: model.detect.no != nc + 4*reg_max"
            
        # 打印损失函数的超参数
        if hasattr(criterion, 'hyp'):
            hyp_dict = criterion.hyp.__dict__ if hasattr(criterion.hyp, '__dict__') else criterion.hyp
            logger.info(f"损失函数超参数: {hyp_dict}")
            
            # 确保criterion.hyp.reg_max与model.detect.reg_max一致
            if hasattr(criterion.hyp, 'reg_max') and hasattr(model, 'detect'):
                if criterion.hyp.reg_max != model.detect.reg_max:
                    logger.warning(f"损失函数reg_max与模型不一致: criterion.hyp.reg_max={criterion.hyp.reg_max}, model.detect.reg_max={model.detect.reg_max}")
                    criterion.hyp.reg_max = model.detect.reg_max
                    logger.info(f"已修复: 设置criterion.hyp.reg_max={criterion.hyp.reg_max}")
        else:
            logger.warning("损失函数没有hyp属性")
            
        # 检查其他关键部分
        if hasattr(criterion, 'no') and hasattr(model, 'detect'):
            if criterion.no != model.detect.no:
                logger.warning(f"损失函数no与模型不一致: criterion.no={criterion.no}, model.detect.no={model.detect.no}")
                criterion.no = model.detect.no
                logger.info(f"已修复: 设置criterion.no={criterion.no}")
                
        if hasattr(criterion, 'nc') and hasattr(model, 'detect'):
            if criterion.nc != model.detect.nc:
                logger.warning(f"损失函数nc与模型不一致: criterion.nc={criterion.nc}, model.detect.nc={model.detect.nc}")
                criterion.nc = model.detect.nc
                logger.info(f"已修复: 设置criterion.nc={criterion.nc}")
                
        if hasattr(criterion, 'reg_max') and hasattr(model, 'detect'):
            if criterion.reg_max != model.detect.reg_max:
                logger.warning(f"损失函数reg_max与模型不一致: criterion.reg_max={criterion.reg_max}, model.detect.reg_max={model.detect.reg_max}")
                criterion.reg_max = model.detect.reg_max
                logger.info(f"已修复: 设置criterion.reg_max={criterion.reg_max}")
                
    except Exception as e:
        logger.error(f"初始化损失函数失败: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        # 仅作为后备方案尝试其他损失函数
        criterion = E2EDetectLoss(model)
        logger.info("使用E2EDetectLoss作为备选损失函数")

    # 优化器设置
    # 为冻结和非冻结参数创建不同的参数组，提高训练效率
    if training_config.get('freeze_backbone', False):
        # 参数分组 - 冻结backbone时，为backbone参数设置lr=0
        pg0, pg1, pg2 = [], [], []  # 优化器参数组
        for k, v in model.named_modules():
            if 'backbone' in k:
                # backbone参数不需要优化，直接跳过所有相关参数
                continue
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # 偏置
            elif isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # 无衰减
            else:
                if hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # 有衰减
                
        initial_lr = model_config.get('lr0', 0.01)
        logger.info(f"使用初始学习率: {initial_lr}")
        
        optimizer = optim.SGD(pg0, lr=initial_lr, momentum=model_config.get('momentum', 0.937), nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': model_config.get('weight_decay', 0.0005)})  # 添加带衰减的参数
        optimizer.add_param_group({'params': pg2, 'weight_decay': 0.0})  # 添加偏置，无权重衰减
        del pg0, pg1, pg2
    else:
        # 标准优化器设置
        # 设置初始学习率
        initial_lr = model_config.get('lr0', 0.01)
        logger.info(f"使用初始学习率: {initial_lr}")
        
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=model_config.get('momentum', 0.937),
                            weight_decay=model_config.get('weight_decay', 0.0005), nesterov=True)
                          
    # 设置batch_size和梯度累积
    batch_size = 4
    grad_accumulation = 4  # 确保逻辑批大小为16
    logger.info(f"使用batch_size: {batch_size}, 梯度累积: {grad_accumulation}, 逻辑批大小: {batch_size * grad_accumulation}")
    
    # 学习率预热设置
    warmup_epochs = training_config.get('warmup_epochs', 10)  # 增加预热轮数
    warmup_bias_lr = training_config.get('warmup_bias_lr', 0.05)  # 降低预热学习率
    final_lr = model_config.get('lr0', 0.005)  # 降低预热后的目标学习率
    logger.info(f"启用学习率预热: {warmup_epochs}轮, 从{initial_lr}到{final_lr}")
    
    # 创建学习率调度器
    scheduler = MultiStepLR(
        optimizer,
        milestones=model_config.get('steps', [100, 150]),
        gamma=model_config.get('gamma', 0.1)
    )
    scaler = amp.GradScaler()
    
    # 加载恢复训练检查点
    start_epoch, best_map = 0, 0.0
    if training_config.get('resume', False):
        ckpt_path = save_dir / 'last.pt'
        if ckpt_path.exists():
            logger.info(f'恢复训练: {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            
            # 加载模型权重
            model.load_state_dict(ckpt['model'])
            
            # 恢复训练状态
            start_epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            scheduler.load_state_dict(ckpt['scheduler'])
            best_map = ckpt.get('best_map', 0.0)
            
            logger.info(f'已恢复训练: epoch {start_epoch}, 最佳mAP {best_map:.4f}')
    
    # 冻结指定层
    freeze_layers = set(training_config.get('freeze', []))
    if len(freeze_layers) > 0:
        logger.info(f"冻结层: {freeze_layers}")
        for i, param in enumerate(model.parameters()):
            param.requires_grad = i not in freeze_layers
    
    # 训练循环
    logger.info(f"开始训练: {training_config.get('epochs', 200)} 轮, {len(train_loader)} 批次/轮")
    
    # 初始化EMA为None
    ema = None
    
    for epoch in range(start_epoch, training_config.get('epochs', 200)):
        model.train()
        
        # 在指定epoch解冻backbone
        if training_config.get('freeze_backbone', False) and epoch >= training_config.get('unfreeze_epoch', 1):
            logger.info(f"已到第{epoch}轮，解冻backbone")
            model = freeze_backbone(model, freeze=False)
            # 重新设置优化器，为所有参数更新梯度
            optimizer = optim.SGD(model.parameters(), lr=model_config.get('lr0', 0.01), momentum=model_config.get('momentum', 0.937),
                               weight_decay=model_config.get('weight_decay', 0.0005))
            logger.info(f"已重新初始化优化器，包含所有模型参数")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{training_config.get('epochs', 200)}")
        mloss = torch.zeros(3, device=device)
        total_targets = 0  # 记录总目标数
        
        # 梯度累积设置
        optimizer.zero_grad(set_to_none=True)
        
        for i,(batch_index,batch,*_) in enumerate(pbar):
            # 准备数据
            imgs = batch['img'].to(device, non_blocking=True).float() / 255.0  # 归一化到[0,1]
            
            # 自动混合精度
            with amp.autocast():
                preds = model(imgs)
                # 将原始 batch 传递给 compute_loss
                loss, loss_items = compute_loss(preds, batch, criterion, model.hyp if isinstance(model.hyp, SimpleNamespace) else model_config)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:  # 如果loss是多维张量
                    loss = loss.mean()  # 取平均值使其成为标量
                loss = loss / grad_accumulation  # 根据梯度累积缩放损失
            
            # 检测NaN并跳过
            if torch.isnan(loss).any():  # 使用.any()来处理多元素张量
                logger.warning(f"检测到NaN损失，跳过此批次更新")
                continue
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 记录总目标数，使用批次中的图像数量
            total_targets += batch['img'].shape[0]
            
            # 梯度累积
            if (i + 1) % grad_accumulation == 0:
                # 梯度裁剪，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                if ema:
                    ema.update(model)
            
            # 更新平均损失
            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix(box_loss=f'{mloss[0]:.4f}', obj_loss=f'{mloss[1]:.4f}', cls_loss=f'{mloss[2]:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        if epoch % 10 == 0 or epoch == training_config.get('epochs', 200) - 1:
            # 评估模型
            results = validate(model, val_loader, model_config, device)
            
            # 保存最佳模型
            mAP50 = results.get('metrics/mAP50', 0.0)
            if mAP50 > best_map:
                best_map = mAP50
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_map': best_map,
                    'config': config,
                }, save_dir / 'best.pt')
                logger.info(f'保存最佳模型: mAP50 {best_map:.4f}')
            
            # 打印验证结果
            logger.info(f"Epoch {epoch}的验证结果: {results}")
        
        # 保存最后一个检查点
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_map': best_map,
            'config': config,
        }, save_dir / 'last.pt')
    
    # 训练完成
    logger.info(f"训练完成: 最佳mAP50 {best_map:.4f}")
    
    # 导出模型
    export_model(model, model_config.get('input_size', 640), save_dir / 'best.pt', save_dir / 'model.onnx')
    
    return save_dir / 'best.pt'

def compute_loss(predictions, current_batch, loss_criterion, hyp_dict_for_fallback):
    """
    计算检测损失，使用Ultralytics的损失函数
    
    参数:
        predictions: 模型预测输出
        current_batch: 从 DataLoader 加载的原始批次数据字典
        loss_criterion: 初始化的损失函数对象 (DetectionLoss 实例)
        hyp_dict_for_fallback: 包含超参数的字典，用于在 loss_criterion.hyp 损坏时回退
        
    返回:
        loss: 总损失 (一个标量张量)
        loss_items: 各项损失 (box, obj, cls) 用于日志记录，形状为 [3]
    """
    # 预测来自于 model(imgs)，其中 imgs 已经在目标设备上。
    # 因此，predictions[0].device 将给出目标设备。
    if not predictions or not isinstance(predictions[0], torch.Tensor):
        logger.warning("Predictions 张量为空或非张量，回退设备的确定可能不可靠。")
        device_for_fallback_tensors = torch.device('cpu')
        for p_tensor_check in predictions:
            if isinstance(p_tensor_check, torch.Tensor):
                device_for_fallback_tensors = p_tensor_check.device
                break
    else:
        device_for_fallback_tensors = predictions[0].device
        
    # 检查预测张量是否正确
    if len(predictions) > 0 and isinstance(predictions[0], torch.Tensor):
        first_pred = predictions[0]
        # 检查是否有NaN值
        if torch.isnan(first_pred).any():
            logger.warning(f"预测张量包含NaN值!")
        else:
            # 打印一些基本信息
            pred_stats = {
                "形状": [p.shape for p in predictions],
                "均值": [p.mean().item() for p in predictions],
                "标准差": [p.std().item() for p in predictions],
                "最小值": [p.min().item() for p in predictions],
                "最大值": [p.max().item() for p in predictions]
            }
            # logger.info(f"预测统计信息: {pred_stats}")

    try:
        # === 调试和修复 loss_criterion.hyp ===
        if not hasattr(loss_criterion, 'hyp') or isinstance(loss_criterion.hyp, dict):
            logger.warning(f"compute_loss: loss_criterion.hyp 是字典或不存在。尝试从 hyp_dict_for_fallback 恢复。类型: {type(loss_criterion.hyp) if hasattr(loss_criterion, 'hyp') else 'N/A'}")
            
            # 确保 hyp_dict_for_fallback 首先是 SimpleNamespace (如果它是字典)
            if isinstance(hyp_dict_for_fallback, dict):
                hyp_dict_for_fallback = SimpleNamespace(**hyp_dict_for_fallback)
                # logger.info("compute_loss: hyp_dict_for_fallback (原为dict) 已转换为 SimpleNamespace")

            required_keys = ['box', 'cls', 'dfl', 'label_smoothing', 'fl_gamma', 'reg_max'] # 根据 DetectionLoss 的需要添加, reg_max 也需要
            if all(hasattr(hyp_dict_for_fallback, key) for key in required_keys):
                loss_criterion.hyp = hyp_dict_for_fallback # 直接使用转换后的 SimpleNamespace
                # logger.info("compute_loss: 已从 hyp_dict_for_fallback (现为SimpleNamespace) 赋值给 loss_criterion.hyp")
            elif hasattr(loss_criterion, 'model') and hasattr(loss_criterion.model, 'hyp') and isinstance(loss_criterion.model.hyp, SimpleNamespace):
                logger.warning("compute_loss: hyp_dict_for_fallback 不完整，尝试从 loss_criterion.model.hyp 恢复 (如果它是 SimpleNamespace)")
                loss_criterion.hyp = loss_criterion.model.hyp
                # logger.info("compute_loss: 已从 loss_criterion.model.hyp 恢复 loss_criterion.hyp")
            else:
                default_hyp_values = {
                    'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'label_smoothing': 0.0, 'fl_gamma': 0.0, 'reg_max': 16 # 修改reg_max为16而不是15
                }
                loss_criterion.hyp = SimpleNamespace(**default_hyp_values)
                logger.warning(f"compute_loss: 已创建包含默认值的最小 loss_criterion.hyp: {loss_criterion.hyp}")
        elif not isinstance(loss_criterion.hyp, SimpleNamespace):
             logger.warning(f"compute_loss: loss_criterion.hyp 不是 SimpleNamespace，类型为 {type(loss_criterion.hyp)}。将尝试强制转换为 SimpleNamespace.")
             try:
                loss_criterion.hyp = SimpleNamespace(**vars(loss_criterion.hyp)) # 如果它是一个具有 __dict__ 的对象
             except TypeError:
                logger.error(f"compute_loss: 无法将 loss_criterion.hyp ({type(loss_criterion.hyp)}) 转换为 SimpleNamespace。保持原样。")

        # 再次检查并记录类型，确保修复生效
        # if not isinstance(loss_criterion.hyp, SimpleNamespace):
        #     logger.error(f"compute_loss: 修复后 loss_criterion.hyp 仍然不是 SimpleNamespace，类型: {type(loss_criterion.hyp)}. 这可能会导致错误。")
        # === 调试和修复结束 ===

        # 检查当前批次的标签数据
        if 'cls' in current_batch and 'bboxes' in current_batch:
            cls = current_batch['cls']
            bboxes = current_batch['bboxes']
            
            if len(cls) > 0:
                        # 直接传递预测列表给损失函数
                loss, loss_items = loss_criterion(predictions, current_batch)
        
        # 数值稳定性检查 - 使用any()检查多元素张量
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("损失为NaN或Inf，使用小的常数损失代替")
            loss = torch.tensor(1e-3, device=device_for_fallback_tensors, requires_grad=True)
            loss_items = torch.tensor([1e-3, 1e-3, 1e-3], device=device_for_fallback_tensors)
    
    except Exception as e:
        logger.error(f"计算损失时出错: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        # 所有方法都失败，使用最简单的备用损失
        # device_for_fallback_tensors 已在上面定义

        if isinstance(predictions, list) and all(isinstance(p, torch.Tensor) for p in predictions) and len(predictions) > 0:
            loss = torch.tensor(0.0, requires_grad=True, device=device_for_fallback_tensors)
            for p_tensor in predictions: # p_tensor 已经位于 device_for_fallback_tensors
                loss = loss + p_tensor.sum() * 0.0001
        elif isinstance(predictions, torch.Tensor): # 处理 predictions 是单个张量的情况
            loss = torch.tensor(0.0, requires_grad=True, device=device_for_fallback_tensors)
            loss = loss + predictions.sum() * 0.0001 # predictions 已经位于 device_for_fallback_tensors
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device_for_fallback_tensors)
            
        loss_items = torch.tensor([0.0, 0.0, 0.0], device=device_for_fallback_tensors)
        
    return loss, loss_items

# import torch
# import torch.nn.functional as F
# from ultralytics.utils.ops import non_max_suppression
# from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
@torch.inference_mode()
def validate(model, dataloader, hyp, device):
    """
    纯 PyTorch + Ultralytics 的验证函数
    -----------------------------------
    支持 DyHead (anchor-free, DFL) 输出，返回常用检测指标字典。
    """
    model.eval()
    # 降低置信度阈值以获取更多候选框
    conf_thres = hyp.get("conf_thres", 0.01)  # 从0.25降低到0.01
    iou_thres  = hyp.get("iou_thres", 0.45)   # 从0.7降低到0.45，与YOLOv8默认值一致
    max_det    = hyp.get("max_det",   300)

    # 初始化度量器
    metrics = DetMetrics(save_dir=None)
    cfm     = ConfusionMatrix(nc=model.detect.nc)
    iouv    = torch.linspace(0.5, 0.95, 10, device=device)
    stats   = []

    proj = torch.arange(model.detect.reg_max, device=device).float()          # DFL 投影向量
    stride = model.stride.to(device)                                          # [8,16,32]

    for batch_idx, batch in enumerate(dataloader):
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.      # [B,C,H,W]
        bs   = imgs.size(0)

        # targets: (image_idx, class, x, y, w, h) —— (xywh 已经归一化)
        targets = torch.cat(
            (batch["batch_idx"].view(-1,1),
             batch["cls"].view(-1,1),
             batch["bboxes"]), dim=1).to(device)

        raw_preds = model(imgs)  # DyHead 输出 list[bs,no,ny,nx]

        # ----------------------------------------------------
        # 1) 将 DyHead 输出解码为每张图一张 [n,6] Tensor (xyxy+conf+cls)
        # ----------------------------------------------------
        decoded_imgs = [[] for _ in range(bs)]

        for level, p in enumerate(raw_preds):            # 对每个特征层
            b, no, ny, nx = p.shape
            p = p.view(b, no, ny, nx)                    # [bs, nc+4*reg_max, ny, nx]

            cls_logits = p[:, :model.detect.nc]          # [bs,nc,ny,nx]
            dfl_logits = p[:, model.detect.nc:]          # [bs,4*reg_max,ny,nx]
            dfl_logits = dfl_logits.view(b, 4, model.detect.reg_max, ny, nx)

            # ---------- DFL 解码 ----------
            dfl_probs = F.softmax(dfl_logits, dim=2)     # softmax over bins
            distances = (dfl_probs * proj.view(1,1,-1,1,1)).sum(2)  # [bs,4,ny,nx]

            # 网格
            yv, xv = torch.meshgrid(torch.arange(ny, device=device),
                                    torch.arange(nx, device=device), indexing="ij")
            grid = torch.stack((xv, yv), 0).float()      # [2,ny,nx]

            # 安全性检查
            if torch.isnan(distances).any() or torch.isinf(distances).any():
                logger.warning(f"检测到NaN/Inf距离值! 级别{level}")
                distances = torch.clamp(distances, 0, 100)  # 应对NaN/Inf

            # 中心点 + 距离，转 xyxy
            xy = (grid + 0.5).unsqueeze(0) * stride[level]  # [1,2,ny,nx]
            wh = distances * stride[level]                  # [bs,4,ny,nx]

            # 更安全的边界框计算方式
            x1 = (xy[:,0] - wh[:,0]).clamp(0)  # 左
            y1 = (xy[:,1] - wh[:,2]).clamp(0)  # 上
            x2 = (xy[:,0] + wh[:,1]).clamp(0)  # 右
            y2 = (xy[:,1] + wh[:,3]).clamp(0)  # 下

            boxes = torch.stack((x1,y1,x2,y2), 1)  # [bs,4,ny,nx]

            # --------- 置信度与类别 ---------
            cls_probs = cls_logits.sigmoid()             # [bs,nc,ny,nx]
            obj_scores = cls_probs.sum(1)                # [bs,ny,nx]
            cls_scores, cls_idx = cls_probs.max(1)       # [bs,ny,nx]

            # 展平为 [bs,N,6]
            boxes   = boxes.permute(0,2,3,1).reshape(b, -1, 4)  # [b,ny*nx,4]
            scores  = cls_scores.reshape(b, -1, 1)              # [b,ny*nx,1]
            obj_scores = obj_scores.reshape(b, -1, 1)           # [b,ny*nx,1]
            cls_idx = cls_idx.reshape(b, -1, 1).float()         # [b,ny*nx,1]

            for i in range(bs):
                final_scores = scores[i].clamp(0, 1.0)  # 使用cls_scores作为最终置信度
                decoded_imgs[i].append(torch.cat((boxes[i], final_scores, cls_idx[i]), 1))

        # 合并三个层级
        decoded_imgs = [torch.cat(img_preds, 0) for img_preds in decoded_imgs]  # len=bs

        # ----------------------------------------------------
        # 2) NMS & 指标累积
        # ----------------------------------------------------
        n_pred_total = 0
        n_matched_total = 0
        
        nms_conf_thres = conf_thres  # 0.01或更低

        for img_i, preds in enumerate(decoded_imgs):
            if len(preds) == 0:
                continue
            
            # 确保预测框在图像范围内
            h, w = imgs.shape[2], imgs.shape[3]
            preds[:, 0].clamp_(0, w)  # x1
            preds[:, 1].clamp_(0, h)  # y1
            preds[:, 2].clamp_(0, w)  # x2
            preds[:, 3].clamp_(0, h)  # y2
            
            # 只保留有效的框
            valid_mask = (preds[:, 2] > preds[:, 0]) & (preds[:, 3] > preds[:, 1])
            if not valid_mask.all():
                preds = preds[valid_mask]
            
            # 应用NMS
            det = non_max_suppression(preds.unsqueeze(0),
                                   nms_conf_thres, iou_thres,
                                   max_det=max_det,
                                   nc=model.detect.nc)[0]
                                   
            n_pred_total += len(det)

            # confusion matrix
            gt = targets[targets[:,0] == img_i]
            if len(gt) or len(det):
                cfm.process_batch(det, xywh2xyxy(gt[:,2:6]), gt[:,1].long())

            # stats 用于 AP 计算
            if len(det) and len(gt):
                predn = det.clone()
                labels = gt.clone()
                nl = len(labels)
                tcls = labels[:, 1].tolist() if nl else []
                
                if nl:
                    labels_xywh_normalized = labels[:, 2:6]
                    labels_xyxy_normalized = xywh2xyxy(labels_xywh_normalized)
                    
                    img_h, img_w = imgs.shape[2], imgs.shape[3]
                    
                    tbox = labels_xyxy_normalized.clone()
                    tbox[:, [0, 2]] *= img_w
                    tbox[:, [1, 3]] *= img_h
                    
                    tbox = tbox.clamp_min(0)
                    
                    correct = torch.zeros((len(predn), len(iouv)), dtype=torch.float32, device=device)
                    
                    if len(predn):
                        pred_boxes = predn[:, :4].clone()
                        pred_boxes[:, 0].clamp_(0, img_w)
                        pred_boxes[:, 1].clamp_(0, img_h)
                        pred_boxes[:, 2].clamp_(0, img_w)
                        pred_boxes[:, 3].clamp_(0, img_h)
                        
                        valid_mask = (pred_boxes[:, 2] > pred_boxes[:, 0]) & (pred_boxes[:, 3] > pred_boxes[:, 1])
                        if not valid_mask.all():
                            if valid_mask.any():
                                pred_boxes = pred_boxes[valid_mask]
                                predn = predn[valid_mask]
                            else:
                                continue
                        
                        pred_cls = predn[:, 5].long()
                        pred_confs = predn[:, 4]
                        
                        iou = torchvision.ops.box_iou(tbox, pred_boxes)
                        
                        for j, iou_thres_value in enumerate(iouv):
                            match_iou = iou >= iou_thres_value
                            
                            if match_iou.numel() == 0:
                                continue
                                
                            for gt_idx in range(len(labels)):
                                gt_class = int(labels[gt_idx, 1])
                                matching_pred_mask = match_iou[gt_idx]
                                
                                if matching_pred_mask.sum() > 0:
                                    matching_preds = torch.nonzero(matching_pred_mask).squeeze(1)
                                    iou_values = iou[gt_idx, matching_preds]
                                    best_idx = matching_preds[iou_values.argmax()]
                                    
                                    if not correct[best_idx, j]:
                                        correct[best_idx, j] = 1.0
                                        n_matched_total += 1
                    
                    stats.append((correct.cpu().float(),
                                 predn[:, 4].cpu().float(),
                                 predn[:, 5].cpu().float(),
                                 torch.tensor(tcls).float() if len(tcls) else torch.zeros(0)))

    # ----------------------------------------------------
    # 3) 计算最终指标
    # ----------------------------------------------------
    try:
        if not stats:
            logger.warning("没有统计数据进行评估")
            return {
                "metrics/precision": 0.0,
                "metrics/recall": 0.0,
                "metrics/mAP50": 0.0,
                "metrics/mAP50-95": 0.0,
                "confusion_matrix": cfm.matrix.cpu().numpy()
            }
            
        processed_stats = []
        for stat_idx, stat_type in enumerate(zip(*stats)):
            if not stat_type:
                processed_stats.append(torch.zeros(0))
                continue
                
            if stat_idx == 0:
                cat_tensor = torch.cat([t.float() for t in stat_type], 0)
                processed_stats.append(cat_tensor)
            else:
                cat_tensor = torch.cat([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in stat_type], 0)
                processed_stats.append(cat_tensor)
                
        metrics.process(*processed_stats)
        
        results = {
            "metrics/precision": metrics.box.p,
            "metrics/recall": metrics.box.r,
            "metrics/mAP50": metrics.box.map50,
            "metrics/mAP50-95": metrics.box.map
        }
        
    except Exception as e:
        logger.error(f"计算指标时出错: {str(e)}")
        logger.exception("异常详情")
        results = {
            "metrics/precision": 0.0,
            "metrics/recall": 0.0,
            "metrics/mAP50": 0.0,
            "metrics/mAP50-95": 0.0
        }
    
    try:
        if isinstance(cfm.matrix, torch.Tensor):
            results["confusion_matrix"] = cfm.matrix.cpu().numpy()
        else:
            results["confusion_matrix"] = cfm.matrix
    except Exception as e:
        logger.warning(f"无法添加混淆矩阵：{str(e)}")
        results["confusion_matrix"] = np.zeros((model.detect.nc, model.detect.nc))
    
    return results



def ap_per_class(tp, conf, pred_cls, target_cls):
    """计算每个类别的平均精度
    
    参数:
        tp: 预测为正确的标志 (N, 10) - 10个IoU阈值
        conf: 置信度分数 (N,)
        pred_cls: 预测类别 (N,)
        target_cls: 真实类别列表 [...]
        
    返回:
        ap: 各类别在各IoU阈值下的AP值 (nc, 10)
        ap_class: 唯一的类别索引
    """
    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数
    
    # 创建矩阵存储AP结果 (nc, 10)
    ap = np.zeros((nc, tp.shape[1]))
    
    for ci, c in enumerate(unique_classes):
        # 获取属于该类别的预测结果
        i = pred_cls == c
        n_p = i.sum()  # 预测数量
        n_gt = (target_cls == c).sum()  # 真值数量
        
        if n_p == 0 or n_gt == 0:
            continue
            
        # 累积TP和FP
        fpc = (1 - tp[i]).cumsum(axis=0)
        tpc = tp[i].cumsum(axis=0)
        
        # 计算召回率和精确率
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)
        
        # 计算平均精度 - 使用所有点方法
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
            
    return ap, unique_classes.astype(int)

def compute_ap(recall, precision):
    """使用11点插值法计算AP值
    
    参数:
        recall: 召回率列表
        precision: 精确率列表
        
    返回:
        AP值
    """
    # 添加起始和结束点
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # 计算精确率包络线
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # 计算召回率区间的差值
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def export_model(model, img_size, weights_path, output_path):
    """
    导出模型为ONNX格式
    
    参数:
        model: 训练好的模型
        img_size: 图像尺寸
        weights_path: 权重文件路径
        output_path: 输出ONNX文件路径
    """
    try:
        import onnx
        
        # 加载最佳权重
        ckpt = torch.load(weights_path)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        try:
            # 尝试使用动态批次
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=False,
                opset_version=11,  # 使用更广泛兼容的版本
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"模型已导出为ONNX格式: {output_path} (opset_version=11, 动态批次)")
        except Exception as e:
            # 如果动态批次失败，尝试使用固定批次
            logger.warning(f"动态批次ONNX导出失败: {e}，尝试固定批次...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=False,
                opset_version=11,
                input_names=['images'],
                output_names=['output']
            )
            logger.info(f"模型已导出为ONNX格式: {output_path} (opset_version=11, 固定批次)")
        
        # 验证ONNX模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return True
    except Exception as e:
        logger.error(f"导出ONNX失败: {e}")
        return False

def xywh2xyxy(x):
    """将边界框从[x, y, w, h]格式转换为[x1, y1, x2, y2]格式
    
    参数:
        x: 边界框张量 [..., 4]
        
    返回:
        转换后的边界框张量
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y + h/2
    return y

if __name__ == '__main__':
    # 设置日志记录级别
    # 减少日志量
    import logging
    logging.getLogger().setLevel(logging.INFO)  # 改为INFO，确保能看到足够的信息
    
    opt = parse_args()
    
    # 只有在verbose模式下才使用详细日志
    if opt.get('verbose', False):
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置混合精度训练
    torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试以提高速度
    
    # 运行训练
    train(opt) 