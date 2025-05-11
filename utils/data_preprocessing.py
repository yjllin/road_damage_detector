"""
数据预处理模块 - 用于YOLOv8s_v8s模型
包含数据集准备和超参数创建功能
"""

import os
import yaml
import shutil
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_custom_hyp(save_path='custom_files/hyp_v8s.yaml'):
    """
    创建优化的超参数文件，用于YOLOv8s_v8s模型训练
    
    参数:
        save_path: 保存路径，默认为'custom_files/hyp_v8s.yaml'
        
    返回:
        保存的超参数文件路径
    """
    hyp = {
        # 训练超参数
        "lr0": 0.01,  # 初始学习率
        "lrf": 0.1,   # 最终学习率 = lr0 * lrf (余弦退火)
        "momentum": 0.937,  # 优化器动量
        "weight_decay": 0.0005,  # 权重衰减
        "warmup_epochs": 3.0,  # 预热epochs
        "warmup_momentum": 0.8,  # 预热初始动量
        "warmup_bias_lr": 0.1,  # 预热初始偏置学习率
        
        # 损失函数权重
        "box": 0.05,  # 边界框损失权重
        "cls": 0.3,   # 分类损失权重
        "cls_pw": 1.0,  # 分类BCELoss正样本权重
        "obj": 0.7,   # 目标检测损失权重
        "obj_pw": 1.0,  # 目标BCELoss正样本权重
        "iou_t": 0.2,  # IoU训练阈值
        "anchor_t": 4.0,  # 锚点损失阈值
        "fl_gamma": 0.0,  # 焦点损失gamma
        
        # 数据增强参数 - 针对小目标道路损伤做了调整
        "hsv_h": 0.015,  # HSV色调增强
        "hsv_s": 0.7,    # HSV饱和度增强
        "hsv_v": 0.4,    # HSV亮度增强
        "degrees": 0.5,  # 旋转角度 (+/- deg)
        "translate": 0.1,  # 平移 (+/- fraction)
        "scale": 0.6,    # 缩放 (+/- gain)
        "shear": 0.2,    # 剪切 (+/- deg)
        "perspective": 0.0,  # 透视变换
        "flipud": 0.1,   # 上下翻转概率
        "fliplr": 0.5,   # 左右翻转概率
        "mosaic": 1.0,   # mosaic数据增强概率
        "mixup": 0.2,    # mixup数据增强概率
        "copy_paste": 0.1,  # 分割粘贴增强
        
        # 特殊参数 - 针对路面损伤检测的优化
        "overlap_mask": True,  # 重叠掩码处理
        "mask_ratio": 4,      # 掩码下采样比例
        "dropout": 0.1,       # 在Detect-Dyhead中使用的dropout率
        "val_conf": 0.001,    # 验证时的置信度阈值
        "nms_conf": 0.25,     # NMS置信度阈值
        "nms_iou": 0.45,      # NMS IoU阈值
    }
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存超参数
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(hyp, f, sort_keys=False)
    
    logger.info(f"超参数已保存到: {save_path}")
    return save_path

def download_file(url, path, chunk_size=1024):
    """
    下载文件并显示进度条
    
    参数:
        url: 下载URL
        path: 保存路径
        chunk_size: 每次下载的块大小
    """
    try:
        # 获取文件大小
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 下载文件
        with open(path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=os.path.basename(path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False

def extract_zip(zip_path, extract_to=None):
    """
    解压ZIP文件
    
    参数:
        zip_path: ZIP文件路径
        extract_to: 解压目标路径，默认为ZIP文件所在目录
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取总文件数
            total = len(zip_ref.namelist())
            
            # 解压文件并显示进度
            for i, file in enumerate(zip_ref.namelist()):
                zip_ref.extract(file, extract_to)
                if i % 100 == 0 or i == total - 1:
                    logger.info(f"解压进度: {i+1}/{total}")
        
        logger.info(f"文件已解压到: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"解压失败: {e}")
        return False

def prepare_dataset(dataset_dir, download_if_missing=True):
    """
    准备道路损伤数据集
    
    参数:
        dataset_dir: 数据集保存目录
        download_if_missing: 如果数据集不存在，是否下载
    
    返回:
        数据集配置文件路径
    """
    dataset_dir = Path(dataset_dir)
    dataset_yaml = dataset_dir / 'dataset.yaml'
    
    # 检查数据集是否存在
    if not dataset_dir.exists() or not (dataset_dir / 'images').exists():
        if not download_if_missing:
            logger.error(f"数据集不存在: {dataset_dir}")
            return None
        
        # 创建数据集目录
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载数据集（以中国道路损伤数据集为例）
        # 注意：这里使用的URL需要根据实际情况替换
        download_url = "https://github.com/example/road_damage_dataset/archive/refs/heads/main.zip"
        zip_path = dataset_dir / "road_damage_dataset.zip"
        
        logger.info(f"下载数据集: {download_url}")
        if not download_file(download_url, zip_path):
            logger.error("数据集下载失败")
            return None
        
        # 解压数据集
        logger.info("解压数据集...")
        if not extract_zip(zip_path, dataset_dir):
            logger.error("数据集解压失败")
            return None
        
        # 删除ZIP文件
        os.remove(zip_path)
    
    # 检查并处理数据集目录结构
    if not dataset_yaml.exists():
        # 创建数据集配置文件
        dataset_config = {
            'path': str(dataset_dir),
            'train': str(dataset_dir / 'images/train'),
            'val': str(dataset_dir / 'images/val'),
            'nc': 4,  # 默认4个类别：裂缝、坑洼、填补、井盖
            'names': ['裂缝', '坑洼', '填补', '井盖']
        }
        
        # 检查目录结构
        for dir_path in [dataset_dir / 'images/train', dataset_dir / 'images/val',
                         dataset_dir / 'labels/train', dataset_dir / 'labels/val']:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        with open(dataset_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, sort_keys=False)
    
    # 验证数据集
    train_images = list((dataset_dir / 'images/train').glob('*.*'))
    val_images = list((dataset_dir / 'images/val').glob('*.*'))
    train_labels = list((dataset_dir / 'labels/train').glob('*.txt'))
    val_labels = list((dataset_dir / 'labels/val').glob('*.txt'))
    
    logger.info(f"数据集统计:")
    logger.info(f"  训练图像: {len(train_images)}张")
    logger.info(f"  验证图像: {len(val_images)}张")
    logger.info(f"  训练标签: {len(train_labels)}个")
    logger.info(f"  验证标签: {len(val_labels)}个")
    
    return str(dataset_yaml)

def convert_labelme_to_yolo(labelme_dir, output_dir, class_mapping=None):
    """
    将LabelMe格式的标注转换为YOLO格式
    
    参数:
        labelme_dir: LabelMe标注文件目录
        output_dir: 输出目录
        class_mapping: 类别映射字典，默认为None
    """
    try:
        import json
        
        if class_mapping is None:
            class_mapping = {
                '裂缝': 0, 
                '坑洼': 1, 
                '填补': 2, 
                '井盖': 3
            }
        
        labelme_files = list(Path(labelme_dir).glob('*.json'))
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"转换{len(labelme_files)}个LabelMe标注为YOLO格式")
        
        for json_file in tqdm(labelme_files):
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像尺寸
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            # 准备输出文件
            output_file = Path(output_dir) / f"{json_file.stem}.txt"
            
            # 转换标注
            with open(output_file, 'w', encoding='utf-8') as f:
                for shape in data['shapes']:
                    # 获取类别
                    class_name = shape['label']
                    if class_name not in class_mapping:
                        logger.warning(f"未知类别: {class_name}, 跳过")
                        continue
                    
                    class_id = class_mapping[class_name]
                    
                    # 获取坐标点
                    points = shape['points']
                    if shape['shape_type'] == 'rectangle':
                        # 矩形: [x1, y1, x2, y2] -> [center_x, center_y, width, height]
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # 计算中心点和尺寸
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = abs(x2 - x1) / img_width
                        height = abs(y2 - y1) / img_height
                        
                        # 写入YOLO格式
                        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
                    elif shape['shape_type'] == 'polygon':
                        # 多边形: 找到外接矩形
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        # 计算中心点和尺寸
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # 写入YOLO格式
                        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
        
        logger.info(f"转换完成，结果保存到: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False

def split_dataset(image_dir, label_dir, output_base_dir, split_ratio=0.8, seed=42):
    """
    分割数据集为训练集和验证集
    
    参数:
        image_dir: 图像目录
        label_dir: 标签目录
        output_base_dir: 输出基础目录
        split_ratio: 训练集比例
        seed: 随机种子
    """
    import random
    import shutil
    from glob import glob
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    train_img_dir = os.path.join(output_base_dir, 'images/train')
    val_img_dir = os.path.join(output_base_dir, 'images/val')
    train_label_dir = os.path.join(output_base_dir, 'labels/train')
    val_label_dir = os.path.join(output_base_dir, 'labels/val')
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(glob(os.path.join(image_dir, f'*.{ext}')))
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 分割数据集
    num_train = int(len(image_files) * split_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]
    
    logger.info(f"数据集分割: 总共{len(image_files)}张图像")
    logger.info(f"  训练集: {len(train_files)}张")
    logger.info(f"  验证集: {len(val_files)}张")
    
    # 复制文件
    for file_list, img_dir, lbl_dir in [
        (train_files, train_img_dir, train_label_dir),
        (val_files, val_img_dir, val_label_dir)
    ]:
        for img_path in file_list:
            # 复制图像
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(img_dir, img_name))
            
            # 复制对应标签
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(lbl_dir, label_name))
    
    # 创建数据集配置文件
    yaml_path = os.path.join(output_base_dir, 'dataset.yaml')
    dataset_config = {
        'path': output_base_dir,
        'train': train_img_dir,
        'val': val_img_dir,
        'nc': 4,
        'names': ['裂缝', '坑洼', '填补', '井盖']
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    
    logger.info(f"数据集已分割，配置文件保存到: {yaml_path}")
    return yaml_path

def check_dataset_balance(label_dir):
    """
    检查数据集类别平衡性
    
    参数:
        label_dir: 标签目录
    """
    import numpy as np
    from collections import Counter
    
    class_counts = Counter()
    label_files = list(Path(label_dir).glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    # 打印类别统计
    logger.info("数据集类别统计:")
    total = sum(class_counts.values())
    for class_id, count in sorted(class_counts.items()):
        percentage = count / total * 100 if total > 0 else 0
        logger.info(f"  类别 {class_id}: {count} 个目标 ({percentage:.2f}%)")
    
    # 计算类别平衡性
    counts = np.array(list(class_counts.values()))
    if len(counts) > 0:
        imbalance = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float('inf')
        logger.info(f"类别不平衡度: {imbalance:.2f}")
        
        if imbalance > 1.0:
            logger.warning("警告: 数据集类别不平衡，可能需要使用类别权重或数据增强")
    
    return dict(class_counts)

# 主功能演示
if __name__ == "__main__":
    # 创建超参数
    hyp_path = create_custom_hyp()
    
    # 准备数据集
    dataset_path = prepare_dataset('database/China_Drone')
    
    # 检查数据集平衡性
    if dataset_path:
        check_dataset_balance('database/China_Drone/labels/train')