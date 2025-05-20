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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
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
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

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

def verify_dataset(data_path):
    """
    验证数据集的完整性
    
    参数:
        data_path: 数据集路径
        
    返回:
        is_valid: 数据集是否有效
        info: 数据集信息字典
    """
    try:
        # 检查data.yaml文件
        yaml_path = Path(data_path) / 'data.yaml'
        if not yaml_path.exists():
            return False, {"error": f"data.yaml不存在: {yaml_path}"}
            
        # 读取配置
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)
            
        # 检查必需的字段
        required_fields = ['path', 'train_images', 'val_images', 'names']
        missing_fields = [field for field in required_fields if field not in data_dict]
        if missing_fields:
            return False, {"error": f"data.yaml缺少必需字段: {missing_fields}"}
            
        # 解析路径 - 处理相对路径
        data_root = Path(data_dict['path'])
        if str(data_root).startswith('./'):
            # 将相对路径从数据集根目录解析
            data_root = Path(data_path).parent / str(data_root)[2:]
            
        # 解析训练和验证集路径
        train_images_path = data_dict['train_images']
        if train_images_path.startswith('./'):
            train_images_path = train_images_path[2:]
        train_path = data_root / train_images_path
        
        val_images_path = data_dict['val_images']
        if val_images_path.startswith('./'):
            val_images_path = val_images_path[2:]
        val_path = data_root / val_images_path
        
        if not train_path.exists():
            return False, {"error": f"训练集路径不存在: {train_path}"}
        if not val_path.exists():
            return False, {"error": f"验证集路径不存在: {val_path}"}
            
        # 检查类别名称
        if not isinstance(data_dict['names'], (list, dict)):
            return False, {"error": "类别名称必须是列表或字典"}
            
        # 统计图像和标签文件 - 使用递归搜索
        train_images = []
        val_images = []

        # 支持的图像格式
        img_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        # 递归搜索训练集图像
        for fmt in img_formats:
            train_images.extend(list(train_path.rglob(fmt)))

        # 递归搜索验证集图像
        for fmt in img_formats:
            val_images.extend(list(val_path.rglob(fmt)))

        # 打印找到的图像文件数量
        logger.info(f"找到训练图像: {len(train_images)}张, 验证图像: {len(val_images)}张")
        
        # 检查对应的标签文件
        train_labels = []
        val_labels = []
        missing_train_labels = []
        missing_val_labels = []
        
        # 获取标签目录路径 (将'images'替换为'labels')
        train_label_path = str(train_path).replace('images', 'labels')
        val_label_path = str(val_path).replace('images', 'labels')
        
        # 确保目录路径是Path对象
        train_label_path = Path(train_label_path)
        val_label_path = Path(val_label_path)
        
        for img_path in train_images:
            # 构建对应的标签文件路径
            label_path = train_label_path / f"{img_path.stem}.txt"
            
            if label_path.exists():
                train_labels.append(label_path)
            else:
                missing_train_labels.append(img_path.name)
                
        for img_path in val_images:
            # 构建对应的标签文件路径
            label_path = val_label_path / f"{img_path.stem}.txt"
            
            if label_path.exists():
                val_labels.append(label_path)
            else:
                missing_val_labels.append(img_path.name)
        
        # 检查标签文件格式
        invalid_labels = []
        for label_path in train_labels + val_labels:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:  # class x y w h
                            invalid_labels.append(label_path.name)
                            break
                        class_id = int(parts[0])
                        if not (0 <= class_id < len(data_dict['names'])):
                            invalid_labels.append(label_path.name)
                            break
            except:
                invalid_labels.append(label_path.name)
        
        # 汇总信息
        info = {
            "train_images": len(train_images),
            "train_labels": len(train_labels),
            "val_images": len(val_images),
            "val_labels": len(val_labels),
            "classes": len(data_dict['names']),
            "missing_train_labels": missing_train_labels,
            "missing_val_labels": missing_val_labels,
            "invalid_labels": invalid_labels
        }
        
        # 判断数据集是否可用
        is_valid = (len(train_images) > 0 and 
                   len(val_images) > 0 and 
                   len(missing_train_labels) == 0 and 
                   len(missing_val_labels) == 0 and 
                   len(invalid_labels) == 0)
        
        return is_valid, info
        
    except Exception as e:
        return False, {"error": f"验证数据集时出错: {str(e)}"}

def create_dataloaders(data_paths, img_size=640, batch_size=4, workers=4, global_config=None):
    """
    创建训练和验证数据加载器，支持多数据集合并
    
    参数:
        data_paths: 数据集路径列表
        img_size: 图像尺寸
        batch_size: 批次大小
        workers: 数据加载的工作线程数
        global_config: 全局配置（可选）
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 类别数量
        class_frequencies: 训练集中各类别的频率
    """
    try:
        # 内存管理
        import psutil
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        logger.info(f"系统内存: 总计={total_memory/1024**3:.1f}GB, 可用={available_memory/1024**3:.1f}GB")
        
        # 根据可用内存调整batch_size和workers
        if available_memory < 8 * 1024**3:  # 如果可用内存小于8GB
            original_batch_size = batch_size
            original_workers = workers
            batch_size = min(batch_size, 8)  # 限制batch_size
            workers = min(workers, 4)        # 限制workers
            if batch_size != original_batch_size or workers != original_workers:
                logger.warning(f"可用内存较低，自动调整参数: batch_size={batch_size} (原{original_batch_size}), "
                             f"workers={workers} (原{original_workers})")
        
        # 验证和规范化数据路径
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        data_paths = [str(Path(p).resolve()) for p in data_paths]  # 转换为绝对路径
        logger.info(f"准备加载多个数据集: {data_paths}")
        
        # 验证数据增强参数
        if global_config and 'data_augmentation' in global_config:
            aug_params = global_config['data_augmentation']
            # 检查必需的参数
            required_params = {
                'mosaic': 0.7,      # 默认值
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'cutout': 0.1
            }
            
            for param, default_value in required_params.items():
                if param not in aug_params:
                    logger.warning(f"数据增强参数 {param} 未设置，使用默认值 {default_value}")
                    aug_params[param] = default_value
            
            # 验证参数值的合理性
            if aug_params.get('mosaic', 0) > 0.9:
                logger.warning("Mosaic概率过高可能影响训练稳定性，建议设置在0.7左右")
            if aug_params.get('degrees', 0) > 20:
                logger.warning("旋转角度过大可能导致目标特征丢失，建议不超过20度")
            if aug_params.get('scale', 0) < 0.1:
                logger.warning("缩放比例过小可能导致目标特征丢失")
        
        # 验证图像尺寸
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        img_size = int(img_size)
        if img_size % 32 != 0:
            new_size = ((img_size + 31) // 32) * 32
            logger.warning(f"图像尺寸应该是32的倍数，自动调整: {img_size} -> {new_size}")
            img_size = new_size
            
        # 验证batch_size
        if batch_size < 1:
            logger.warning(f"batch_size不能小于1，自动设置为1")
            batch_size = 1
        
        # 验证workers
        if workers < 0:
            logger.warning(f"workers不能为负数，自动设置为0")
            workers = 0
        
        # 单一路径情况处理
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        logger.info(f"准备加载多个数据集: {data_paths}")
        
        # 合并数据集
        merged_train_ds = None
        merged_val_ds = None
        class_names = None
        num_classes = 0
        
        # 准备数据增强参数
        augment_params = SimpleNamespace(
            # 基础几何变换
            degrees=15.0,  # 增加旋转角度范围
            translate=0.2,  # 增加平移比例
            scale=0.7,    # 增加缩放比例
            shear=0.3,    # 增加剪切角度
            perspective=0.001,  # 添加透视变换
            flipud=0.2,   # 增加上下翻转概率
            fliplr=0.5,   # 保持左右翻转概率
            
            # 马赛克和混合增强
            mosaic=0.8,  # 增加Mosaic概率
            mixup=0.3,   # 增加Mixup概率
            copy_paste=0.2,  # 增加Copy-paste概率
            copy_paste_mode="flip",  # Copy-paste模式
            copy_paste_iou_thresh=0.3,  # Copy-paste IoU阈值
            copy_paste_min_area=0.3,  # Copy-paste最小区域比例
            cutmix=0.2,   # 添加CutMix增强
            
            # 颜色空间增强
            hsv_h=0.015,  # HSV色调增强
            hsv_s=0.7,    # HSV饱和度增强
            hsv_v=0.4,    # HSV亮度增强
            bgr=True,     # 使用BGR格式
            hue=0.015,    # 色调调整
            saturation=0.7,  # 饱和度调整
            value=0.4,    # 明度调整
            augment=True,  # 启用增强
            
            # 图像质量增强
            blur=0.02,    # 增加模糊概率
            gray=0.02,    # 增加灰度概率
            contrast=0.5,  # 增加对比度调整
            brightness=0.3,  # 增加亮度调整
            
            # 遮挡和掩码增强
            cutout=0.2,   # 增加Cutout概率
            overlap_mask=True,  # 掩码重叠
            mask_ratio=4,  # 掩码比例
            
            # 图像和标签处理
            h=640,        # 图像高度
            w=640,        # 图像宽度
            scale_min=0.5,  # 最小缩放
            scale_max=1.5,  # 最大缩放
            stride=32,    # 步长
            pad=0.0,      # 填充
            rect=False,   # 矩形训练
            label_smoothing=0.1,  # 增加标签平滑系数
            
            # 训练控制参数
            cache=False,  # 是否缓存图像
            image_weights=False,  # 是否使用图像权重
            val=False,    # 是否是验证模式
            seed=0,       # 随机种子
            fraction=1.0,  # 数据集比例
            single_cls=False,  # 是否是单类别
            rect_training=False,  # 是否使用矩形训练
            cos_lr=False,  # 是否使用余弦学习率
            close_mosaic=10,  # 关闭马赛克的轮数
            resume=False,  # 是否恢复训练
            amp=True,     # 是否使用混合精度训练
            workers=8,  # 数据加载线程数
            quad=False,  # 是否使用四元组
            prefix="",    # 前缀
            
            # 验证集参数
            val_rect=False,  # 验证集是否使用矩形训练
            val_pad=0.5,    # 验证集填充比例
            val_stride=32,  # 验证集步长
            val_scale=None,  # 验证集缩放比例
            
            # 其他必需参数
            dropout=0.1,  # 增加dropout比例
            channels=3,   # 输入通道数
            normalized=True,  # 是否归一化
            auto_augment=True,  # 启用自动增强
            rect_mode=False,  # 矩形训练模式
            cache_images=False,  # 是否缓存图像
            save_dir=None,  # 保存目录
            verbose=False  # 是否显示详细信息
        )
        
        # 如果有全局配置中的数据增强参数，更新默认值
        if global_config and 'data_augmentation' in global_config:
            for k, v in global_config['data_augmentation'].items():
                if hasattr(augment_params, k):
                    setattr(augment_params, k, v)
        
        # 验证每个数据集
        valid_data_paths = []
        for data_path in data_paths:
            is_valid, info = verify_dataset(data_path)
            if is_valid:
                logger.info(f"数据集验证通过: {data_path}")
                logger.info(f"- 训练集: {info['train_images']}张图片, {info['train_labels']}个标签")
                logger.info(f"- 验证集: {info['val_images']}张图片, {info['val_labels']}个标签")
                logger.info(f"- 类别数: {info['classes']}")
                valid_data_paths.append(data_path)
            else:
                if "error" in info:
                    logger.error(f"数据集验证失败: {data_path} - {info['error']}")
                else:
                    if info['missing_train_labels']:
                        count = len(info['missing_train_labels'])
                        logger.error(f"训练集缺少标签文件: {count}个文件")
                        if count > 0 and count <= 5:
                            logger.error(f"缺失文件: {info['missing_train_labels']}")
                        elif count > 5:
                            logger.error(f"部分缺失文件: {info['missing_train_labels'][:5]}... (共{count}个)")
                    if info['missing_val_labels']:
                        count = len(info['missing_val_labels'])
                        logger.error(f"验证集缺少标签文件: {count}个文件")
                        if count > 0 and count <= 5:
                            logger.error(f"缺失文件: {info['missing_val_labels']}")
                        elif count > 5:
                            logger.error(f"部分缺失文件: {info['missing_val_labels'][:5]}... (共{count}个)")
                    if info['invalid_labels']:
                        count = len(info['invalid_labels'])
                        logger.error(f"无效的标签文件: {count}个文件")
                        if count > 0 and count <= 5:
                            logger.error(f"无效文件: {info['invalid_labels']}")
                        elif count > 5:
                            logger.error(f"部分无效文件: {info['invalid_labels'][:5]}... (共{count}个)")
        
        if not valid_data_paths:
            raise ValueError("没有可用的数据集")
            
        data_paths = valid_data_paths
        
        for data_path in data_paths:
            # 加载配置
            data_yaml = os.path.join(data_path, 'data.yaml')
            
            # 读取配置
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_dict = yaml.safe_load(f)
                
            # 检查类别是否一致
            if class_names is None:
                class_names = data_dict.get('names', [])
                num_classes = len(class_names)
            else:
                current_names = data_dict.get('names', [])
                if len(current_names) != num_classes or any(a != b for a, b in zip(class_names, current_names)):
                    logger.warning(f"数据集 {data_path} 的类别与之前的数据集不一致，可能导致问题")
            
            data_dict.setdefault('channels', 3)
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            
            # 创建数据集 - 处理相对路径
            data_root = Path(data_dict['path'])
            if str(data_root).startswith('./'):
                # 将相对路径从数据集根目录解析
                data_root = Path(data_path).parent / str(data_root)[2:]

            # 解析训练和验证集路径
            train_images_path = data_dict['train_images']
            if train_images_path.startswith('./'):
                train_images_path = train_images_path[2:]
            train_path = str(data_root / train_images_path)

            val_images_path = data_dict['val_images']
            if val_images_path.startswith('./'):
                val_images_path = val_images_path[2:]
            val_path = str(data_root / val_images_path)

            # 确保路径存在
            if not os.path.exists(train_path):
                logger.warning(f"训练集路径不存在: {train_path}")
                continue
                
            if not os.path.exists(val_path):
                logger.warning(f"验证集路径不存在: {val_path}")
                continue

            # 获取标签路径 - 确保与验证函数一致
            train_label_path = train_path.replace('images', 'labels')
            val_label_path = val_path.replace('images', 'labels')

            # 确保标签目录存在
            if not os.path.exists(train_label_path):
                logger.warning(f"训练集标签路径不存在: {train_label_path}")
                continue
                
            if not os.path.exists(val_label_path):
                logger.warning(f"验证集标签路径不存在: {val_label_path}")
                continue
            
            logger.info(f"加载数据集 {data_path}...")
            
            try:
                # 创建数据集，启用增强的数据增强
                train_ds = SafeYOLODataset(
                    img_path=train_path, 
                    data=data_dict, 
                    imgsz=img_size, 
                    augment=True, 
                    cache=False,
                    hyp=augment_params,  # 传递增强参数
                    prefix=f"{data_path}: "  # 添加数据集前缀
                )
                
                val_ds = SafeYOLODataset(
                    img_path=val_path, 
                    data=data_dict, 
                    imgsz=img_size, 
                    augment=False, 
                    cache=False,
                    hyp=augment_params,  # 验证集不会使用增强，但仍需传递参数
                    prefix=f"{data_path}: "  # 添加数据集前缀
                )
                
                # 验证数据集是否正确加载
                if not hasattr(train_ds, 'im_files') or len(train_ds.im_files) == 0:
                    logger.error(f"训练集 {data_path} 加载失败: 没有找到图像文件")
                    continue
                    
                if not hasattr(val_ds, 'im_files') or len(val_ds.im_files) == 0:
                    logger.error(f"验证集 {data_path} 加载失败: 没有找到图像文件")
                    continue
                
                # 验证图像和标签数量是否匹配
                if len(train_ds.im_files) != len(train_ds.labels):
                    logger.error(f"训练集 {data_path} 的图像数量({len(train_ds.im_files)})与标签数量({len(train_ds.labels)})不匹配")
                    continue
                    
                if len(val_ds.im_files) != len(val_ds.labels):
                    logger.error(f"验证集 {data_path} 的图像数量({len(val_ds.im_files)})与标签数量({len(val_ds.labels)})不匹配")
                    continue
                
                # 合并数据集
                if merged_train_ds is None:
                    merged_train_ds = train_ds
                    merged_val_ds = val_ds
                else:
                    # 使用安全的合并方法
                    merged_train_ds.extend(train_ds)
                    merged_val_ds.extend(val_ds)
                
                # 校验代码（确保数据集长度一致性）
                logger.info(f"训练集: {len(merged_train_ds.im_files)} 张图片, {len(merged_train_ds.ims)} 个缓存")
                logger.info(f"验证集: {len(merged_val_ds.im_files)} 张图片, {len(merged_val_ds.ims)} 个缓存")
                
            except Exception as e:
                logger.error(f"加载数据集 {data_path} 时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        if merged_train_ds is None or merged_val_ds is None:
            raise ValueError("没有成功加载任何数据集")
        
        # 计算训练集中的类别频率
        class_frequencies = [0] * num_classes
        if hasattr(merged_train_ds, 'labels') and merged_train_ds.labels:
            for label_data_for_image in merged_train_ds.labels:
                if 'cls' in label_data_for_image and isinstance(label_data_for_image['cls'], torch.Tensor):
                    cls_tensor = label_data_for_image['cls'].squeeze().long()
                    if cls_tensor.ndim == 0:
                        cls_tensor = cls_tensor.unsqueeze(0)
                    
                    for cls_idx in cls_tensor:
                        class_id = cls_idx.item()
                        if 0 <= class_id < num_classes:
                            class_frequencies[class_id] += 1
            logger.info(f"计算得到的训练集类别频率: {class_frequencies}")
        else:
            logger.warning("merged_train_ds.labels 不可用或为空, 无法计算类别频率。")
            class_frequencies = [1] * num_classes  # 默认每个类别频率为1
        
        # 为每个样本计算权重用于重采样
        from torch.utils.data import WeightedRandomSampler
        
        # 为每个样本计算权重 (类别频率的倒数)
        sample_weights = []
        for label_data in merged_train_ds.labels:
            if 'cls' in label_data and isinstance(label_data['cls'], torch.Tensor):
                cls_tensor = label_data['cls'].squeeze().long()
                if cls_tensor.ndim == 0:
                    cls_tensor = cls_tensor.unsqueeze(0)
                
                # 计算该图像的权重 (使用最高频率类别的倒数)
                img_weight = 0
                for cls_idx in cls_tensor:
                    class_id = cls_idx.item()
                    if 0 <= class_id < len(class_frequencies):
                        # 使用类别频率的倒数作为权重
                        weight = 1.0 / (class_frequencies[class_id] + 1e-6)
                        img_weight = max(img_weight, weight)  # 使用最大权重
                
                sample_weights.append(img_weight)
            else:
                sample_weights.append(1.0)  # 默认权重
        
        # 创建 WeightedRandomSampler
        if len(sample_weights) > 0:
            logger.info(f"使用 WeightedRandomSampler 进行类别平衡")
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            # 使用torch.utils.data.DataLoader而不是build_dataloader
            train_loader = torch.utils.data.DataLoader(
                merged_train_ds,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sampler,
                pin_memory=True,
                collate_fn=getattr(merged_train_ds, 'collate_fn', None)
            )
        else:
            logger.warning("无法创建采样器，使用普通随机采样")
            train_loader = torch.utils.data.DataLoader(
                merged_train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
                collate_fn=getattr(merged_train_ds, 'collate_fn', None)
            )
            
        # 验证集不需要重采样
        val_loader = torch.utils.data.DataLoader(
            merged_val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=getattr(merged_val_ds, 'collate_fn', None)
        )
        
        logger.info(f"数据加载器创建成功:")
        logger.info(f"- 合并后的训练集: {merged_train_ds.ni} 张图片")
        logger.info(f"- 合并后的验证集: {merged_val_ds.ni} 张图片")
        logger.info(f"- 类别数: {num_classes}")
        logger.info(f"- 图像尺寸: {img_size}")
        logger.info(f"- 批次大小: {batch_size}")
        
        # 检查数据加载器的大小和批次大小
        logger.info(f"训练数据批次数: {len(train_loader)}, 批次大小: {train_loader.batch_size}")
        logger.info(f"验证数据批次数: {len(val_loader)}, 批次大小: {val_loader.batch_size}")
        
        # 添加对dataset的检查
        if hasattr(train_loader.dataset, 'ensure_cache_lists'):
            logger.info("重新检查训练数据集缓存列表一致性...")
            train_loader.dataset.ensure_cache_lists()
        if hasattr(val_loader.dataset, 'ensure_cache_lists'):
            logger.info("重新检查验证数据集缓存列表一致性...")
            val_loader.dataset.ensure_cache_lists()
        
        return train_loader, val_loader, num_classes, class_frequencies
        
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        logger.error(f"数据路径: {data_paths}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
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

class ModelEMA:
    """模型指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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

    # 初始化模型
    logger.info(f"初始化模型")
    model = ImprovedYoloV8s()
    model = model.to(device)
    
    # 执行sanity-check
    def sanity_check():
        logger.info("执行 sanity-check...")
        with torch.no_grad():
            feats = model(torch.randn(1,3,640,640).to(device))
            print(f"特征图形状: {[f.shape for f in feats]}")   # → [(1,70,80,80), (1,70,40,40), (1,70,20,20)]
            # 通道验证
            assert feats[0][:,64:].std() > 0, "分类通道无活动"  # cls
            assert feats[0][:,:64].std() > 0, "回归通道无活动"  # dfl
            logger.info("所有检查通过!")
    
    sanity_check()
    
    # 创建输出目录
    save_dir = Path(training_config.get('output', 'runs/train')) / training_config.get('name', 'exp')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件处理器
    log_file = add_file_handler(save_dir)
    logger.info(f"日志文件: {log_file}")

    # 获取多数据集路径
    data_paths = data_config.get('paths', ['./database/China_MotorBike'])
    if isinstance(data_paths, str):
        data_paths = [data_paths]  # 确保是列表
    
    logger.info(f"使用数据集路径: {data_paths}")
    
    # 添加数据集预检查
    logger.info("执行数据集预检查...")
    for data_path in data_paths:
        try:
            # 验证数据路径
            if not os.path.exists(data_path):
                logger.error(f"数据集路径不存在: {data_path}")
                continue
                
            # 检查data.yaml文件
            yaml_path = os.path.join(data_path, 'data.yaml')
            if not os.path.exists(yaml_path):
                logger.error(f"data.yaml文件不存在: {yaml_path}")
                continue
                
            # 读取数据集配置
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_dict = yaml.safe_load(f)
                
            # 检查必需的字段
            required_fields = ['path', 'train_images', 'val_images', 'names']
            missing_fields = [field for field in required_fields if field not in data_dict]
            if missing_fields:
                logger.error(f"数据集 {data_path} 的data.yaml缺少必需字段: {missing_fields}")
                continue
                
            # 解析路径
            data_root_path = data_dict['path']
            if data_root_path.startswith('./'):
                # 相对路径处理
                data_root_path = os.path.join(os.path.dirname(data_path), data_root_path[2:])
            
            # 检查路径存在性
            if not os.path.exists(data_root_path):
                logger.error(f"数据集根目录不存在: {data_root_path}")
                continue
                
            # 检查训练集路径
            train_images_path = data_dict['train_images']
            if train_images_path.startswith('./'):
                train_images_path = train_images_path[2:]
            train_path = os.path.join(data_root_path, train_images_path)
            
            if not os.path.exists(train_path):
                logger.error(f"训练集路径不存在: {train_path}")
                continue
                
            # 检查验证集路径
            val_images_path = data_dict['val_images']
            if val_images_path.startswith('./'):
                val_images_path = val_images_path[2:]
            val_path = os.path.join(data_root_path, val_images_path)
            
            if not os.path.exists(val_path):
                logger.error(f"验证集路径不存在: {val_path}")
                continue
                
            # 检查标签目录
            train_label_path = os.path.join(os.path.dirname(train_path), 'labels')
            val_label_path = os.path.join(os.path.dirname(val_path), 'labels')
            
            if not os.path.exists(train_label_path):
                logger.error(f"训练集标签目录不存在: {train_label_path}")
                continue
                
            if not os.path.exists(val_label_path):
                logger.error(f"验证集标签目录不存在: {val_label_path}")
                continue
                
            # 简单检查图像文件数量
            train_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                train_images.extend(list(Path(train_path).glob(ext)))
            
            val_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                val_images.extend(list(Path(val_path).glob(ext)))
                
            if len(train_images) == 0:
                logger.error(f"训练集中没有找到图像文件: {train_path}")
                continue
                
            if len(val_images) == 0:
                logger.error(f"验证集中没有找到图像文件: {val_path}")
                continue
                
            # 数据集通过预检查
            logger.info(f"数据集预检查通过: {data_path}")
            logger.info(f"- 训练集图像: {len(train_images)}张")
            logger.info(f"- 验证集图像: {len(val_images)}张")
            logger.info(f"- 类别数: {len(data_dict['names'])}")
            
        except Exception as e:
            logger.error(f"数据集 {data_path} 预检查出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 创建数据加载器
    train_loader, val_loader, num_classes, class_frequencies = create_dataloaders(
        data_paths=data_paths,
        img_size=model_config.get('input_size', 640),
        batch_size=training_config.get('batch_size', 16),
        workers=training_config.get('workers', 8),
        global_config=config
    )
    
    # 计算并设置类别权重
    if 'loss' in config and 'class_weights' in config['loss']:
        class_weights = config['loss']['class_weights']
        logger.info(f"从config.yaml读取类别权重: {class_weights}")
    else:
        # 如果没有配置，则计算类别权重
        if class_frequencies:
            class_weights = []
            sum_freq = sum(class_frequencies)
            for freq in class_frequencies:
                if freq > 0:
                    # 使用1/sqrt(freq)计算权重，平衡小类别
                    class_weights.append(1.0 / math.sqrt(freq))
                else:
                    class_weights.append(1.0) # 对于频率为0的类别，设置默认权重1.0
            
            # 归一化权重
            weight_sum = sum(class_weights)
            class_weights = [w / weight_sum * len(class_weights) for w in class_weights]
            
            # 将计算出的类别权重存入config，以便保存到yaml和后续使用
            loss_config = config.setdefault('loss', {})
            loss_config['class_weights'] = class_weights  # 直接存储为list
            logger.info(f"已计算并存储类别权重到 config['loss']['class_weights']: {class_weights}")
        else:
            logger.warning("未获取到类别频率，跳过类别权重计算。")
            class_weights = None

    # 保存完整配置
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # 更新模型的类别数
    if hasattr(model, 'detect'):
        logger.info(f"更新模型检测头的类别数: {num_classes}")
        if model.detect.nc != num_classes:
            model.detect.nc = num_classes
            logger.info(f"已更新检测头类别数: nc={model.detect.nc}")
    
    # 读取模型的当前配置
    if hasattr(model, 'args'):
        current_args = model.args.__dict__ if hasattr(model.args, '__dict__') else model.args
        logger.info(f"模型当前参数: {current_args}")
    
    # 更新模型参数，使用update而不是覆盖
    if hasattr(model, 'args'):
        if isinstance(model.args, dict):
            model.args.update(model_config)
        else:
            vars(model.args).update(model_config)
    
    # 重新生成 model.hyp 以确保它反映了所有更新，并且是一个 SimpleNamespace
    if hasattr(model, 'args'):
        if isinstance(model.args, dict):
            model.hyp = SimpleNamespace(**model.args)
        else:
            model.hyp = SimpleNamespace(**vars(model.args))
        logger.info("已从合并的 model.args 更新 model.hyp")
        logger.info(f"更新后的模型参数: {model.hyp.__dict__ if hasattr(model.hyp, '__dict__') else model.hyp}")

    # 将 class_weights 添加到 model.hyp
    if class_weights is not None:
        if hasattr(model, 'hyp'):
            # 确保 model.hyp 是 SimpleNamespace，可以动态添加属性
            if not isinstance(model.hyp, SimpleNamespace):
                logger.warning(f"model.hyp 不是 SimpleNamespace (类型: {type(model.hyp)})。尝试将其转换为 SimpleNamespace。")
                try:
                    current_hyp_dict = model.hyp.__dict__ if hasattr(model.hyp, '__dict__') else model.hyp
                    model.hyp = SimpleNamespace(**current_hyp_dict)
                except TypeError:
                    logger.error(f"无法将 model.hyp 转换为 SimpleNamespace。class_weights 可能不会被设置。")
            
            if isinstance(model.hyp, SimpleNamespace): # 再次检查
                # 确保class_weights是list类型
                model.hyp.class_weights = class_weights  # 直接存储为list
                logger.info(f"已将 class_weights 添加到 model.hyp: {model.hyp.class_weights}")
        else:
            logger.warning("model.hyp 未找到, 无法设置 class_weights。")

    # 使用新的compile_model方法编译模型
    if training_config.get('use_compile', False):
        model = model.compile_model()
    
    # 确保检测头的device设置正确
    model.model[-1].device = device
    
    # 初始化损失函数
    try:
        # 确保loss配置存在
        if 'loss' not in config:
            config['loss'] = {}
            
        # 设置默认损失参数
        default_loss_params = {
            'box': 7.5,  # 边界框损失权重
            'cls': 0.5,  # 分类损失权重
            'dfl': 1.5,  # DFL损失权重
            'label_smoothing': 0.0,  # 标签平滑系数
            'fl_gamma': 2.0,  # Focal Loss gamma
            'reg_max': 16  # DFL回归最大值
        }
        
        # 更新配置中的损失参数
        for k, v in default_loss_params.items():
            if k not in config['loss']:
                config['loss'][k] = v
                
        # 将损失配置转换为SimpleNamespace
        loss_hyp = SimpleNamespace(**config['loss'])
        
        # 初始化损失函数
        criterion = DetectionLoss(model)
        criterion.hyp = loss_hyp
        
        logger.info("使用Ultralytics v8DetectionLoss")
        logger.info(f"损失函数配置: {vars(loss_hyp)}")
        
        # 添加断言检查
        assert model.detect.no == 69 and criterion.no == 69, f"模型和损失函数的no必须都等于69: model.detect.no={model.detect.no}, criterion.no={criterion.no}"
        assert model.detect.reg_max == criterion.reg_max, f"模型和损失函数的reg_max不匹配: model.detect.reg_max={model.detect.reg_max}, criterion.reg_max={criterion.reg_max}"
        assert model.detect.nc == criterion.nc, f"模型和损失函数的类别数不匹配: model.detect.nc={model.detect.nc}, criterion.nc={criterion.nc}"
        logger.info("断言检查通过: model.detect 和 criterion 配置一致")
        
        # 检查class_weights长度
        if hasattr(criterion, 'hyp') and hasattr(criterion.hyp, 'class_weights'):
            if isinstance(criterion.hyp.class_weights, (list, tuple)):
                assert len(criterion.hyp.class_weights) == model.detect.nc, \
                    f"class_weights长度({len(criterion.hyp.class_weights)})与类别数({model.detect.nc})不匹配"
                logger.info(f"class_weights长度检查通过: {len(criterion.hyp.class_weights)} == {model.detect.nc}")
        
    except Exception as e:
        logger.error(f"初始化损失函数失败: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise  # 直接抛出异常，不再使用备选损失函数

    # 设置batch_size，取消梯度累积
    batch_size = training_config.get('batch_size', 16)  # 设置默认值为16
    logger.info(f"使用batch_size: {batch_size}")
    
    # 设置初始学习率，根据batch_size线性缩放
    base_batch_size = 64  # 基准batch size
    base_lr = 0.01  # 基准学习率
    linear_scale = batch_size / base_batch_size
    initial_lr = base_lr * linear_scale
    logger.info(f"初始学习率: {initial_lr:.6f} (batch_size={batch_size}, scale={linear_scale:.3f}, base_lr={base_lr})")
    
    # 设置优化器（不冻结backbone）
    optimizer = optim.SGD(
        model.parameters(), 
        lr=float(initial_lr), 
        momentum=float(model_config.get('momentum', 0.937)),
        weight_decay=float(model_config.get('weight_decay', 0.0005)), 
        nesterov=True
    )
    
    # 学习率预热设置
    warmup_epochs = int(training_config.get('warmup_epochs', 3))  # 减少预热轮数
    total_warmup_iters = len(train_loader) * warmup_epochs  # 总预热迭代次数
    logger.info(f"启用学习率预热: {warmup_epochs}轮, 共{total_warmup_iters}次迭代")
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(training_config.get('epochs', 200)) - warmup_epochs,
        eta_min=float(model_config.get('lr0', 0.004)) / 100
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
    
    # 初始化EMA
    ema = ModelEMA(model, decay=0.9999)
    logger.info("已启用模型EMA (decay=0.9999)")
    
    # 训练循环
    logger.info(f"开始训练: {training_config.get('epochs', 200)} 轮, {len(train_loader)} 批次/轮")
    
    # 记录当前迭代次数
    cur_iter = 0
    
    for epoch in range(start_epoch, training_config.get('epochs', 200)):
        model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{training_config.get('epochs', 200)}")
        mloss = torch.zeros(3, device=device)
        total_targets = 0  # 记录总目标数
        
        # 学习率预热
        if epoch < warmup_epochs:
            # 逐batch线性预热，确保第0epoch时学习率不为0
            for param_group in optimizer.param_groups:
                warmup_factor = max(0.1, min(1.0, (cur_iter + 1) / total_warmup_iters))  # 确保最小学习率为初始学习率的10%
                param_group['lr'] = initial_lr * warmup_factor
            if cur_iter % 10 == 0:  # 每10个batch记录一次学习率
                logger.info(f"预热学习率: {optimizer.param_groups[0]['lr']:.6f} (iter={cur_iter}/{total_warmup_iters})")
        else:
            # 使用余弦退火调度器
            scheduler.step()
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            if epoch == warmup_epochs or epoch == training_config.get('epochs', 200) - 1:
                logger.info(f"学习率: {current_lr:.6f}")
        
        # 重置梯度，不再需要梯度累积
        optimizer.zero_grad(set_to_none=True)
        
        try:
            for i,(batch_index,batch,*_) in enumerate(pbar):
                # 更新迭代次数
                cur_iter += 1
                
                # 检查批次的完整性
                if 'img' not in batch or 'cls' not in batch or 'bboxes' not in batch:
                    logger.warning(f"批次 {i} 数据不完整，跳过")
                    continue
                
                # 准备数据
                try:
                    imgs = batch['img'].to(device, non_blocking=True).float() / 255.0  # 归一化到[0,1]
                except Exception as e:
                    logger.error(f"处理图像数据时出错: {str(e)}")
                    continue
                
                # 自动混合精度
                with amp.autocast():
                    try:
                        preds = model(imgs)
                    except Exception as e:
                        logger.error(f"模型前向传播出错: {str(e)}")
                        continue
                    
                    try:
                        # 将原始 batch 传递给 compute_loss
                        loss, loss_items = compute_loss(preds, batch, criterion, model.hyp if isinstance(model.hyp, SimpleNamespace) else model_config)
                        if isinstance(loss, torch.Tensor) and loss.dim() > 0:  # 如果loss是多维张量
                            loss = loss.mean()  # 取平均值使其成为标量
                    except Exception as e:
                        logger.error(f"计算损失时出错: {str(e)}")
                        continue
                
                # 检测NaN并跳过
                if torch.isnan(loss).any():  # 使用.any()来处理多元素张量
                    logger.warning(f"检测到NaN损失，跳过此批次更新")
                    continue
                    
                # 反向传播
                try:
                    scaler.scale(loss).backward()
                except Exception as e:
                    logger.error(f"反向传播出错: {str(e)}")
                    continue
                
                # 记录总目标数，使用批次中的图像数量
                total_targets += batch['img'].shape[0]
                
                # 不再需要梯度累积判断，直接更新
                # 梯度裁剪，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # 更新EMA
                ema.update(model)
                
                # 更新平均损失
                mloss = (mloss * i + loss_items) / (i + 1)
                pbar.set_postfix(box=f'{mloss[0]:.4f}', dfl=f'{mloss[1]:.4f}', cls=f'{mloss[2]:.4f}')
        except Exception as e:
            logger.error(f"训练循环中遇到错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        if epoch % 10 == 0 or epoch == training_config.get('epochs', 200) - 1:
            # 使用EMA模型进行验证
            ema.apply_shadow()
            logger.info(f"开始验证模型 (使用EMA)")
            val_results = validate(
                model, 
                val_loader, 
                model.hyp if isinstance(model.hyp, SimpleNamespace) else SimpleNamespace(**model_config),
                device
            )
            ema.restore()
            
            # 保存最佳模型
            mAP50 = val_results["metrics/mAP50"]
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
            logger.info(f"Epoch {epoch}的验证结果: precision: {val_results['metrics/precision']}, recall: {val_results['metrics/recall']}, mAP50: {val_results['metrics/mAP50']}, mAP50-95: {val_results['metrics/mAP50-95']}")
        
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

def decode_dfl(box_dist, anchors, stride, reg_max=16):
    B, C, N = box_dist.shape
    box_dist = box_dist.view(B, 4, reg_max, N).permute(0,3,1,2)  # [B,N,4,reg_max]
    prob = F.softmax(box_dist, dim=-1)
    bins = torch.arange(reg_max, device=box_dist.device, dtype=torch.float) + 0.5
    offsets = (prob * bins).sum(-1)  # [B,N,4]
    xy = anchors.unsqueeze(0).expand(B,-1,-1)
    x1 = xy[:,:,0] - offsets[:,:,0]
    y1 = xy[:,:,1] - offsets[:,:,1]
    x2 = xy[:,:,0] + offsets[:,:,2]
    y2 = xy[:,:,1] + offsets[:,:,3]
    return torch.stack([x1,y1,x2,y2], dim=-1) * stride.unsqueeze(0).unsqueeze(-1)

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
        raise ValueError("Predictions 张量为空或非张量")
        
    device_for_fallback_tensors = predictions[0].device
        
    # 检查预测张量是否正确
    if len(predictions) > 0 and isinstance(predictions[0], torch.Tensor):
        first_pred = predictions[0]
        # 检查是否有NaN值
        if torch.isnan(first_pred).any():
            raise ValueError("预测张量包含NaN值")
            
        # 打印一些基本信息
        pred_stats = {
            "形状": [p.shape for p in predictions],
            "均值": [p.mean().item() for p in predictions],
            "标准差": [p.std().item() for p in predictions],
            "最小值": [p.min().item() for p in predictions],
            "最大值": [p.max().item() for p in predictions]
        }
        logger.debug(f"预测统计信息: {pred_stats}")

    # === 调试和修复 loss_criterion.hyp ===
    if not hasattr(loss_criterion, 'hyp') or isinstance(loss_criterion.hyp, dict):
        logger.warning(f"compute_loss: loss_criterion.hyp 是字典或不存在。尝试从 hyp_dict_for_fallback 恢复。类型: {type(loss_criterion.hyp) if hasattr(loss_criterion, 'hyp') else 'N/A'}")
        
        # 确保 hyp_dict_for_fallback 首先是 SimpleNamespace (如果它是字典)
        if isinstance(hyp_dict_for_fallback, dict):
            hyp_dict_for_fallback = SimpleNamespace(**hyp_dict_for_fallback)

        required_keys = ['box', 'cls', 'dfl', 'label_smoothing', 'fl_gamma', 'reg_max']
        if all(hasattr(hyp_dict_for_fallback, key) for key in required_keys):
            loss_criterion.hyp = hyp_dict_for_fallback
        elif hasattr(loss_criterion, 'model') and hasattr(loss_criterion.model, 'hyp') and isinstance(loss_criterion.model.hyp, SimpleNamespace):
            loss_criterion.hyp = loss_criterion.model.hyp
        else:
            raise ValueError("无法恢复 loss_criterion.hyp，缺少必要的超参数")
            
    elif not isinstance(loss_criterion.hyp, SimpleNamespace):
        raise TypeError(f"loss_criterion.hyp 不是 SimpleNamespace，类型为 {type(loss_criterion.hyp)}")

    # 检查当前批次的标签数据
    if 'cls' not in current_batch or 'bboxes' not in current_batch:
        raise ValueError("批次数据缺少必要的标签字段")
        
    cls = current_batch['cls']
    bboxes = current_batch['bboxes']
    
    if len(cls) == 0:
        raise ValueError("批次中没有标签数据")
        
    # 直接传递预测列表给损失函数
    loss, loss_items = loss_criterion(predictions, current_batch)

    # 数值稳定性检查
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        raise ValueError("损失为NaN或Inf")
    
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
    确保所有张量都在正确的设备上。
    """
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()
    
    # 使用与训练相同的阈值
    conf_thres = getattr(hyp, "conf_thres", 0.001)  # 从0.25降低到0.001
    iou_thres = getattr(hyp, "iou_thres", 0.5)   # 从0.7降低到0.5，与YOLOv8默认值一致
    max_det = getattr(hyp, "max_det", 300)

    # 初始化度量器
    metrics = DetMetrics(save_dir=None)
    cfm = ConfusionMatrix(nc=model.detect.nc)
    iouv = torch.tensor([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], device=device)
    stats = []

    proj = torch.arange(model.detect.reg_max, device=device).float()  # DFL 投影向量
    stride = model.stride.to(device)
    
    # 在验证前记录设备信息
    logger.info(f"验证设备: {device}")
    logger.info(f"模型设备: {next(model.parameters()).device}")

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

            # 分离预测头输出
            dfl_logits = p[:, :4*model.detect.reg_max]      # 64
            cls_logits = p[:, 4*model.detect.reg_max:]      # 6
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
            y1 = (xy[:,1] - wh[:,1]).clamp(0)  # 上
            x2 = (xy[:,0] + wh[:,2]).clamp(0)  # 右
            y2 = (xy[:,1] + wh[:,3]).clamp(0)  # 下

            boxes = torch.stack((x1,y1,x2,y2), 1)  # [bs,4,ny,nx]

            # --------- 置信度与类别 ---------
            # 直接使用类别得分作为置信度
            cls_scores = cls_logits.sigmoid()             # [bs,nc,ny,nx]
            max_cls_scores, cls_idx = cls_scores.max(1)   # [bs,ny,nx]
            conf = max_cls_scores  # [bs,ny,nx]

            # 展平为 [bs,N,6]
            boxes   = boxes.permute(0,2,3,1).reshape(b, -1, 4)  # [b,ny*nx,4]
            conf    = conf.reshape(b, -1, 1)                    # [b,ny*nx,1]
            cls_idx = cls_idx.reshape(b, -1, 1).float()         # [b,ny*nx,1]

            for i in range(bs):
                final_scores = conf[i].clamp(0, 1.0)  # 使用计算出的conf作为最终置信度
                decoded_imgs[i].append(torch.cat((boxes[i], final_scores, cls_idx[i]), 1))

        # 合并三个层级
        decoded_imgs = [torch.cat(img_preds, 0) for img_preds in decoded_imgs]  # len=bs

        # ----------------------------------------------------
        # 2) NMS & 指标累积
        # ----------------------------------------------------
        n_pred_total = 0
        n_matched_total = 0
        
        nms_conf_thres = conf_thres  # 使用与训练相同的阈值

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
                # 确保det在正确的设备上
                if det.device != device:
                    det = det.to(device)
                
                predn = det.clone()
                labels = gt.clone()
                nl = len(labels)
                tcls = labels[:, 1].tolist() if nl else []
                
                if nl:
                    # 确保这些张量在正确的设备上
                    labels_xywh_normalized = labels[:, 2:6]
                    labels_xyxy_normalized = xywh2xyxy(labels_xywh_normalized)
                    
                    img_h, img_w = imgs.shape[2], imgs.shape[3]
                    
                    tbox = labels_xyxy_normalized.clone()
                    tbox[:, [0, 2]] *= img_w
                    tbox[:, [1, 3]] *= img_h
                    
                    # 使用clamp确保盒子坐标有效
                    tbox = tbox.clamp(min=0)
                    
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
                        
                        # 确保tbox和pred_boxes在同一设备上
                        if tbox.device != pred_boxes.device:
                            tbox = tbox.to(pred_boxes.device)
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
                                 torch.tensor(tcls, device='cpu').float() if len(tcls) else torch.zeros(0, device='cpu')))

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
            
        # 直接使用metrics处理统计数据
        metrics.process(*[torch.cat(x, 0).cpu() for x in zip(*stats)])
        
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
    
    # 绘制PR曲线和混淆矩阵
    if len(stats) > 0 and hasattr(model, 'names'):
        try:
            # 提取必要的数据
            tp_all = torch.cat([t[0] for t in stats], 0).cpu().numpy()
            conf_all = torch.cat([t[1] for t in stats], 0).cpu().numpy()
            pred_cls_all = torch.cat([t[2] for t in stats], 0).cpu().numpy()
            target_cls_all = torch.cat([t[3] for t in stats], 0).cpu().numpy()
            
            # 保存目录
            save_dir = Path('runs/train') / 'plots'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 绘制PR曲线
            plot_pr_curve(tp_all, conf_all, pred_cls_all, target_cls_all, save_dir, model.names)
            
            # 绘制混淆矩阵
            plot_confusion_matrix(results["confusion_matrix"], save_dir, model.names)
            
            logger.info(f"已保存PR曲线和混淆矩阵到 {save_dir}")
        except Exception as e:
            logger.error(f"绘图失败: {e}")
    
    logger.info(f"验证完成: 预测总数={n_pred_total}, 匹配总数={n_matched_total}")
    logger.info(f"验证指标: mAP@0.5={metrics.box.map50:.4f}, mAP@0.5-0.95={metrics.box.map:.4f}")
    
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
        
        # 计算AP (使用所有点方法)
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

def plot_pr_curve(tp, conf, pred_cls, target_cls, save_dir=None, names=None):
    """
    绘制精确率-召回率曲线
    
    参数:
        tp: 预测为正确的标志 [N, 11]
        conf: 置信度分数 [N]
        pred_cls: 预测类别 [N]
        target_cls: 目标类别 [M]
        save_dir: 保存目录
        names: 类别名称
    """
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 确保保存目录存在
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保所有输入都是numpy数组
        tp = np.array(tp)
        conf = np.array(conf)
        pred_cls = np.array(pred_cls).astype(int)  # 确保pred_cls是整数类型
        target_cls = np.array(target_cls).astype(int)  # 确保target_cls是整数类型
        
        # 获取唯一类别
        unique_classes = np.unique(target_cls)
        unique_classes = unique_classes.astype(int)  # 确保unique_classes是整数类型
        nc = len(unique_classes)
        
        if nc == 0:
            return
            
        # 设置图形
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        
        # 按置信度排序
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        
        # 为每个类别绘制PR曲线
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # 真实标签数量
            n_p = i.sum()  # 预测数量
            
            if n_p == 0 or n_l == 0:
                continue
                
            # 累积TP和FP
            fpc = (1 - tp[i]).cumsum(axis=0)
            tpc = tp[i].cumsum(axis=0)
            
            # 计算召回率和精确率
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc)
            
            # 绘制PR曲线
            ax.plot(recall, precision, label=f'{names[c]}' if names else f'class {c}')
            
        # 设置图形属性
        ax.set_xlabel('召回率')
        ax.set_ylabel('精确率')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        
        # 保存图形
        if save_dir is not None:
            plt.savefig(save_dir / 'pr_curve.png', dpi=250, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制PR曲线时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def plot_confusion_matrix(confusion_matrix, save_dir=None, names=None):
    """
    绘制混淆矩阵
    
    参数:
        confusion_matrix: 混淆矩阵 [nc, nc]
        save_dir: 保存目录
        names: 类别名称
    """
    try:
        if confusion_matrix.sum() < 1:
            logger.warning("混淆矩阵为空，跳过绘制")
            return None
            
        # 确保保存目录存在
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        nc = confusion_matrix.shape[0]
        
        # 使用log归一化
        cm = confusion_matrix.copy()
        with np.errstate(divide='ignore'):
            cm = np.log(cm + 1)  # 添加1避免log(0)
        
        # 设置图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        
        # 绘制热图
        im = ax.imshow(cm, cmap='Blues')
        
        # 添加类别标签
        ax.set_xticks(np.arange(nc))
        ax.set_yticks(np.arange(nc))
        
        if names is not None and len(names) == nc:
            ax.set_xticklabels(names)
            ax.set_yticklabels(names)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加计数值
        thresh = cm.max() / 2
        for i in range(nc):
            for j in range(nc):
                if confusion_matrix[i, j] > 0:
                    ax.text(j, i, format(int(confusion_matrix[i, j]), 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_title('混淆矩阵')
        fig.colorbar(im)
        
        # 保存图形
        if save_dir is not None:
            fig.savefig(save_dir / 'confusion_matrix.png', dpi=250)
            plt.close(fig)
        
        return fig
    except Exception as e:
        logger.error(f"绘制混淆矩阵失败: {e}")
        return None

# 在导入部分添加以下内容
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
import copy

# 在现有函数之前，添加自定义数据集类
class SafeYOLODataset(YOLODataset):
    """安全的YOLO数据集，确保索引不会超出范围"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保所有缓存列表都初始化为正确的长度
        self.ensure_cache_lists()
        
    def ensure_cache_lists(self):
        """确保所有缓存列表的长度与im_files一致"""
        n = len(self.im_files)
        
        # 初始化或调整ims列表
        if not hasattr(self, 'ims') or len(self.ims) != n:
            self.ims = [None] * n
            
        # 初始化或调整im_hw0列表
        if not hasattr(self, 'im_hw0') or len(self.im_hw0) != n:
            self.im_hw0 = [None] * n
            
        # 初始化或调整im_hw列表
        if not hasattr(self, 'im_hw') or len(self.im_hw) != n:
            self.im_hw = [None] * n
            
        # 初始化或调整npy_files列表
        if not hasattr(self, 'npy_files') or len(self.npy_files) != n:
            self.npy_files = [None] * n
            
        # 初始化或调整shapes列表
        if not hasattr(self, 'shapes') or len(self.shapes) != n:
            self.shapes = [(self.imgsz, self.imgsz)] * n
            
        # 确保ni属性正确
        self.ni = n
    
    def load_image(self, i):
        """重写load_image方法，确保索引不会超出范围"""
        # 确保索引有效
        if i >= len(self.im_files):
            raise IndexError(f"索引 {i} 超出im_files范围 (长度: {len(self.im_files)})")
        
        # 确保ims等列表已初始化且长度一致
        self.ensure_cache_lists()
        
        # 使用基类的load_image方法
        return super().load_image(i)
    
    # 安全地合并数据集
    def extend(self, other_dataset):
        """安全地扩展数据集"""
        if not isinstance(other_dataset, (YOLODataset, SafeYOLODataset)):
            raise TypeError(f"无法扩展类型为 {type(other_dataset)} 的数据集")
        
        # 记录原始长度
        original_len = len(self.im_files)
        
        # 扩展基础属性
        self.im_files.extend(other_dataset.im_files)
        self.labels.extend(other_dataset.labels)
        
        # 确保other_dataset的缓存列表已初始化
        if isinstance(other_dataset, SafeYOLODataset):
            other_dataset.ensure_cache_lists()
        else:
            # 如果是基础YOLODataset，手动初始化缓存列表
            n = len(other_dataset.im_files)
            if not hasattr(other_dataset, 'ims') or other_dataset.ims is None:
                other_dataset.ims = [None] * n
            if not hasattr(other_dataset, 'im_hw0') or other_dataset.im_hw0 is None:
                other_dataset.im_hw0 = [None] * n
            if not hasattr(other_dataset, 'im_hw') or other_dataset.im_hw is None:
                other_dataset.im_hw = [None] * n
            if not hasattr(other_dataset, 'npy_files') or other_dataset.npy_files is None:
                other_dataset.npy_files = [None] * n
            if not hasattr(other_dataset, 'shapes') or other_dataset.shapes is None:
                other_dataset.shapes = [(self.imgsz, self.imgsz)] * n
        
        # 扩展缓存列表
        self.ims.extend(other_dataset.ims)
        self.im_hw0.extend(other_dataset.im_hw0)
        self.im_hw.extend(other_dataset.im_hw)
        self.npy_files.extend(other_dataset.npy_files)
        self.shapes.extend(other_dataset.shapes)
        
        # 更新图像数量
        self.ni = len(self.im_files)
        
        # 返回扩展的数据集
        return self

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