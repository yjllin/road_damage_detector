import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_xml_to_yolo(xml_file, image_width, image_height, class_map):
    """
    将XML格式的标签转换为YOLOv5格式
    
    Args:
        xml_file: XML文件路径
        image_width: 图像宽度
        image_height: 图像高度
        class_map: 类别映射字典
        
    Returns:
        转换后的YOLOv5格式标签列表
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    for obj in root.findall('object'):
        # 获取类别名称
        class_name = obj.find('name').text
        
        # 如果类别不在映射中，跳过
        if class_name not in class_map:
            continue
            
        class_id = class_map[class_name]
        # 获取边界框坐标
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为YOLOv5格式 (x_center, y_center, width, height)
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # 添加到结果列表
        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
        yolo_lines.append(yolo_line)
    return yolo_lines

def get_image_dimensions(image_path):
    """
    获取图像尺寸
    这里使用简单方法，实际应使用PIL或OpenCV
    
    Returns:
        默认尺寸(宽度, 高度)
    """
    # 注意：这里简化处理，实际应当读取图片获取真实尺寸
    # 需要安装PIL: from PIL import Image
    # img = Image.open(image_path)
    # return img.width, img.height
    
    return 1280, 720  # 假设默认尺寸

def split_dataset(dataset_path, train_ratio=0.8):
    """
    将数据集分割为训练集和验证集
    
    Args:
        dataset_path: 数据集路径
        train_ratio: 训练集比例，默认0.8
    """
    # 确保输入路径存在
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"数据集路径 {dataset_path} 不存在")
        return
    
    # 获取训练集图片和标签路径
    train_img_path = dataset_path / 'train' / 'images'
    train_xml_path = dataset_path / 'train' / 'annotations' / 'xmls'
    
    if not train_img_path.exists() or not train_xml_path.exists():
        print(f"训练集路径不完整: {dataset_path}")
        return
    
    # 创建YOLOv5格式的目录结构
    yolo_train_img_path = dataset_path / 'yolo' / 'train' / 'images'
    yolo_train_label_path = dataset_path / 'yolo' / 'train' / 'labels'
    yolo_val_img_path = dataset_path / 'yolo' / 'val' / 'images'
    yolo_val_label_path = dataset_path / 'yolo' / 'val' / 'labels'
    
    # 创建目录
    yolo_train_img_path.mkdir(parents=True, exist_ok=True)
    yolo_train_label_path.mkdir(parents=True, exist_ok=True)
    yolo_val_img_path.mkdir(parents=True, exist_ok=True)
    yolo_val_label_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = list(train_img_path.glob('*.jpg')) + list(train_img_path.glob('*.png'))
    
    # 随机打乱并选择验证集
    random.seed(42)  # 设置随机种子以确保可重复性
    random.shuffle(image_files)
    val_size = int(len(image_files) * (1 - train_ratio))
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    # 类别映射（根据实际情况调整）
    # 这里假设所有数据集使用相同的类别映射
    # 请根据实际情况修改
    class_map = {
        'D00': 0,  # 纵向裂缝(Longitudinal Cracks)
        'D10': 1,  # 横向裂缝(Transverse Cracks)
        'D20': 2,  # 鳄鱼裂缝(Alligator Cracks)
        'D40': 3,  # 坑洞(Potholes)
        'Repair': 4,  # 修补(Repair)
        'Block Crack': 5  # 块状裂缝(Block Crack)
    }
    
    # 处理训练集文件
    for img_path in train_files:
        # 复制图片
        dst_path = yolo_train_img_path / img_path.name
        shutil.copy(str(img_path), str(dst_path))
        
        # 转换对应的XML标签为YOLOv5格式
        xml_name = img_path.stem + '.xml'
        xml_path = train_xml_path / xml_name
        
        if xml_path.exists():
            # 获取图像尺寸
            width, height = get_image_dimensions(img_path)
            
            # 转换XML为YOLOv5格式
            yolo_lines = convert_xml_to_yolo(xml_path, width, height, class_map)
            
            # 保存YOLOv5格式的标签
            label_path = yolo_train_label_path / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    # 处理验证集文件
    for img_path in val_files:
        # 复制图片
        dst_path = yolo_val_img_path / img_path.name
        shutil.copy(str(img_path), str(dst_path))
        
        # 转换对应的XML标签为YOLOv5格式
        xml_name = img_path.stem + '.xml'
        xml_path = train_xml_path / xml_name
        
        if xml_path.exists():
            # 获取图像尺寸
            width, height = get_image_dimensions(img_path)
            
            # 转换XML为YOLOv5格式
            yolo_lines = convert_xml_to_yolo(xml_path, width, height, class_map)
            
            # 保存YOLOv5格式的标签
            label_path = yolo_val_label_path / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    print(f"数据集 {dataset_path.name} 分割完成:")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")
    print(f"数据已转换为YOLOv5格式，保存在 {dataset_path}/yolo 目录")

def main():
    # 基础路径
    base_path = Path('./database')
    
    # 遍历所有数据集目录
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            print(f"\n处理数据集: {dataset_dir.name}")
            split_dataset(dataset_dir)

if __name__ == '__main__':
    main()