#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进的YOLOv8s模型
"""

import argparse
import torch
import os
import sys
import numpy as np
import cv2
from pathlib import Path
import time
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from improved_yolov8.model import ImprovedYoloV8s

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试改进的YOLOv8s模型')
    parser.add_argument('--model-only', action='store_true', help='仅测试模型结构')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA测试')
    parser.add_argument('--config', type=str, default='improved_yolov8/config.yaml', help='模型配置文件')
    return parser.parse_args()

def test_model_structure(config_path, use_cuda=False):
    """测试模型结构"""
    logger.info("测试模型结构...")
    
    # 设置设备
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化模型
    model = ImprovedYoloV8s(config_path=config_path)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 随机输入前向传播测试
    input_shape = (1, 3, 640, 640)
    logger.info(f"前向传播测试 - 输入形状: {input_shape}")
    
    x = torch.randn(input_shape).to(device)
    
    # 计时前向传播
    start_time = time.time()
    with torch.no_grad():
        outputs = model(x)
    forward_time = time.time() - start_time
    
    # 打印输出信息
    logger.info(f"前向传播时间: {forward_time:.4f}秒")
    logger.info(f"输出层数: {len(outputs)}")
    
    # 打印每层输出形状
    for i, output in enumerate(outputs):
        logger.info(f"输出层 {i}: 形状={output.shape}")
    
    return model

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 测试模型结构
    if args.model_only:
        test_model_structure(args.config, args.cuda)
    else:
        # 测试整个推理流程
        logger.info("使用以下命令测试完整推理流程:")
        logger.info("python improved_yolov8/infer.py --weights runs/train/exp/best.pt --source database/China_MotorBike/test --img-size 640")

if __name__ == "__main__":
    main() 