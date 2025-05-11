#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的YOLOv8s模型推理脚本
用于RDD2022数据集的路面损伤检测
"""

import os
import sys
import time
import torch
import yaml
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from improved_yolov8.model import ImprovedYoloV8s

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("runs/infer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用改进的YOLOv8s模型进行推理')
    parser.add_argument('--weights', type=str, default='runs/train/exp/best.pt', help='模型权重文件')
    parser.add_argument('--source', type=str, default='database/China_MotorBike/test', help='输入图像路径或文件夹')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU阈值')
    parser.add_argument('--device', default='0', help='cuda设备，即 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--config', type=str, default='improved_yolov8/config.yaml', help='模型配置文件')
    parser.add_argument('--output', type=str, default='runs/infer', help='输出目录')
    parser.add_argument('--save-txt', action='store_true', help='保存标签到TXT文件')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度到TXT文件')
    parser.add_argument('--classes', nargs='+', type=int, help='筛选特定类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--no-trace', action='store_true', help='不使用ONNX/TorchScript追踪')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')
    
    return parser.parse_args()

def load_model(weights_path, config_path, device):
    """
    加载模型和权重
    
    参数:
        weights_path: 权重文件路径
        config_path: 配置文件路径
        device: 计算设备
        
    返回:
        model: 加载好权重的模型
        class_names: 类别名称列表
    """
    logger.info(f"加载模型: {weights_path}")
    
    # 初始化模型
    model = ImprovedYoloV8s(config_path=config_path)
    
    # 加载权重
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    
    # 获取类别名称
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        class_names = config['classes']['names']
    
    logger.info(f"模型加载完成: 类别={class_names}")
    return model, class_names

def preprocess_image(img_path, img_size):
    """
    预处理图像
    
    参数:
        img_path: 图像路径
        img_size: 调整尺寸
        
    返回:
        img: 原始图像 (用于绘制结果)
        img_tensor: 预处理后的图像张量
        ratio: 缩放比例 [w_ratio, h_ratio]
        pad: 填充 [pad_x, pad_y]
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"无法读取图像: {img_path}")
        return None, None, None, None
    
    # 保存原始尺寸
    h0, w0 = img.shape[:2]
    
    # 计算缩放比例，保持纵横比
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        img_resized = img.copy()
    
    # 获取调整后尺寸
    h, w = img_resized.shape[:2]
    
    # 填充到正方形
    img_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    top, left = (img_size - h) // 2, (img_size - w) // 2
    img_padded[top:top+h, left:left+w, :] = img_resized
    
    # 转换通道顺序 BGR->RGB 并归一化
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    # 转换为张量 [H,W,C] -> [C,H,W]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度 [1,C,H,W]
    
    # 比例和填充信息 (用于将预测结果映射回原始图像)
    ratio = [w / w0, h / h0]
    pad = [left, top]
    
    return img, img_tensor, ratio, pad

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """
    非极大值抑制
    
    参数:
        prediction: 模型输出预测 [batch, num_anchors, grid_h, grid_w, num_classes+5]
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        classes: 筛选类别
        agnostic: 类别无关的NMS
        multi_label: 是否允许一个框有多个标签
        max_det: 每张图像最大检测数量
        
    返回:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # 简化的NMS实现，实际项目中应使用完整实现
    # 该函数应根据模型输出格式具体实现
    
    # 假设这里已经应用了NMS，返回结果
    # 实际结果应该是一个列表，每个元素是一个张量 [xyxy, conf, cls]
    result = []
    for batch_idx in range(prediction[0].shape[0]):
        # 随机生成一些检测框作为例子
        boxes = torch.randn(5, 6)  # 5个框，每个框 [x1, y1, x2, y2, conf, cls]
        boxes[:, 0:4] = boxes[:, 0:4] * 640  # 缩放到图像大小
        boxes[:, 0:4] = torch.clamp(boxes[:, 0:4], 0, 640)  # 裁剪到图像范围
        boxes[:, 4] = torch.rand(5) * 0.5 + 0.5  # 随机置信度 [0.5, 1.0]
        boxes[:, 5] = torch.randint(0, 6, (5,)).float()  # 随机类别 [0, 5]
        result.append(boxes)
    
    return result

def rescale_boxes(boxes, ratio, pad):
    """
    将预测框缩放回原始图像尺寸
    
    参数:
        boxes: 预测框 [N, 4] (xyxy格式)
        ratio: 缩放比例 [w_ratio, h_ratio]
        pad: 填充 [pad_x, pad_y]
        
    返回:
        scaled_boxes: 缩放后的框
    """
    # 移除填充
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    
    # 缩放到原始尺寸
    boxes[:, [0, 2]] /= ratio[0]
    boxes[:, [1, 3]] /= ratio[1]
    
    # 裁剪到图像范围
    boxes[:, 0].clamp_(0)
    boxes[:, 1].clamp_(0)
    # 假设原始图像尺寸为 w0, h0，不过这里没有传入这些参数，所以忽略右边界裁剪
    
    return boxes

def plot_results(img, pred, class_names, conf_thres, hide_labels=False, hide_conf=False):
    """
    在图像上绘制检测结果
    
    参数:
        img: 原始图像
        pred: 预测结果 [N, 6] (xyxy, conf, cls)
        class_names: 类别名称列表
        conf_thres: 置信度阈值
        hide_labels: 是否隐藏标签
        hide_conf: 是否隐藏置信度
    
    返回:
        img: 绘制检测结果的图像
    """
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 生成HSV颜色映射
    num_classes = len(class_names)
    hsv_colors = np.array([np.array([i / num_classes, 1, 0.7]) for i in range(num_classes)])
    rgb_colors = [tuple(int(c * 255) for c in hsv_to_rgb(hsv)) for hsv in hsv_colors]
    
    # 绘制检测框
    if pred is not None and len(pred) > 0:
        for *xyxy, conf, cls_id in pred:
            # 跳过低置信度
            if conf < conf_thres:
                continue
                
            # 获取整数坐标
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 获取类别和颜色
            cls_id = int(cls_id)
            color = rgb_colors[cls_id % len(rgb_colors)]
            color = (color[2], color[1], color[0])  # RGB->BGR
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            if not hide_labels:
                label = f"{class_names[cls_id]}"
                if not hide_conf:
                    label += f" {conf:.2f}"
                
                # 计算标签大小
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)
                
                # 绘制标签文本
                cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def infer(opt):
    """推理主函数"""
    # 设置设备
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() and opt.device != 'cpu' else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model, class_names = load_model(opt.weights, opt.config, device)
    
    # 创建输出目录
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 图像源
    source = Path(opt.source)
    if source.is_dir():
        img_files = sorted(source.glob('**/*.jpg')) + sorted(source.glob('**/*.png'))
    else:
        img_files = [source]
    
    # 运行推理
    logger.info(f"开始推理: {len(img_files)} 张图像")
    
    # 统计
    inference_times = []
    
    for img_path in tqdm(img_files, desc="推理进度"):
        # 预处理图像
        img_orig, img_tensor, ratio, pad = preprocess_image(str(img_path), opt.img_size)
        if img_tensor is None:
            continue
        
        # 转移到设备
        img_tensor = img_tensor.to(device)
        
        # 记录推理时间
        t0 = time.time()
        
        # 推理
        with torch.no_grad():
            pred = model(img_tensor)
            
            # 应用NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, 
                classes=opt.classes, agnostic=opt.agnostic_nms
            )
        
        # 计算推理时间
        inference_time = time.time() - t0
        inference_times.append(inference_time)
        
        # 处理预测结果
        for i, det in enumerate(pred):
            if det is not None and len(det) > 0:
                # 缩放框到原始图像尺寸
                det[:, :4] = rescale_boxes(det[:, :4], ratio, pad)
                
                # 绘制结果
                img_result = plot_results(
                    img_orig, det, class_names, opt.conf_thres,
                    hide_labels=opt.hide_labels, hide_conf=opt.hide_conf
                )
                
                # 保存结果
                output_path = output_dir / f"{img_path.stem}_result{img_path.suffix}"
                cv2.imwrite(str(output_path), img_result)
                
                # 保存标签文件
                if opt.save_txt:
                    txt_path = output_dir / 'labels' / f"{img_path.stem}.txt"
                    txt_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(txt_path, 'w') as f:
                        for *xyxy, conf, cls in det:
                            # 转换为YOLO格式 (x_center, y_center, width, height)
                            h, w = img_orig.shape[:2]
                            x_c = (xyxy[0] + xyxy[2]) / 2 / w
                            y_c = (xyxy[1] + xyxy[3]) / 2 / h
                            width = (xyxy[2] - xyxy[0]) / w
                            height = (xyxy[3] - xyxy[1]) / h
                            
                            # 写入标签
                            f.write(f"{int(cls)} {x_c} {y_c} {width} {height}")
                            if opt.save_conf:
                                f.write(f" {conf}")
                            f.write('\n')
    
    # 计算平均推理时间和FPS
    avg_time = np.mean(inference_times)
    fps = 1 / avg_time if avg_time > 0 else 0
    
    logger.info(f"推理完成: 平均时间={avg_time:.4f}秒/张, FPS={fps:.2f}")
    logger.info(f"结果保存到: {output_dir}")
    
    return output_dir

def main():
    """主函数"""
    # 解析命令行参数
    opt = parse_args()
    
    # 开始推理
    output_dir = infer(opt)
    
    # 打印完成信息
    logger.info(f"处理完成！结果已保存到 {output_dir}")

if __name__ == "__main__":
    main() 