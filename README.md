# 道路损伤检测器 (Road Damage Detector)

基于改进的YOLOv8模型的道路损伤检测系统，用于自动识别和分类道路表面上的各种损伤类型，如裂缝、坑洼等。

## 项目结构

```
road_damage_detector/
├── improved_yolov8/        # 改进的YOLOv8模型实现
│   ├── model.py           # 模型定义文件
│   ├── train.py           # 训练脚本
│   ├── infer.py           # 推理脚本
│   ├── config.yaml        # 配置文件
│   └── __init__.py
├── utils/                  # 工具函数
│   ├── data_preprocessing.py  # 数据预处理
│   ├── dataset_split.py       # 数据集分割
│   ├── dataset_comprise.py    # 数据集压缩
│   └── detectDataset.py       # 数据集检测
├── database/               # 数据集目录
│   ├── China_Drone/        # 中国无人机数据集
│   ├── China_MotorBike/    # 中国摩托车数据集
│   ├── Japan/              # 日本数据集
│   ├── Norway/             # 挪威数据集
│   └── United_States/      # 美国数据集
├── test_model.py           # 模型测试脚本
└── requirements.txt        # 项目依赖
```

## 主要特点

- 基于YOLOv8的改进模型，专为道路损伤检测优化
- 支持多个国家/地区的道路损伤数据集
- 自定义检测头(DyHead)提高了对小目标的检测能力
- 兼容Ultralytics的标准训练流程和损失函数

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/road_damage_detector.git
cd road_damage_detector
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python improved_yolov8/train.py --config improved_yolov8/config.yaml
```

### 推理/测试

```bash
python test_model.py --weights path/to/your/weights.pt --source path/to/test/images
```

或者：

```bash
python improved_yolov8/infer.py --weights path/to/your/weights.pt --source path/to/test/images
```

## 模型改进

这个项目实现了对标准YOLOv8模型的多项改进：

1. 自定义动态检测头(DyHead)以增强特征表示
2. 改进的特征融合机制
3. 优化的损失函数，与Ultralytics的v8DetectionLoss兼容

## 数据集

本项目使用来自多个国家的道路损伤数据集：
- 中国无人机数据集
- 中国摩托车数据集
- 日本道路损伤数据集
- 挪威道路损伤数据集
- 美国道路损伤数据集

## 许可证

MIT 