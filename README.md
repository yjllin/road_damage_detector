# 改进版 YOLOv8 道路损伤检测器

这个项目是基于 YOLOv8 的改进版本，专门用于道路损伤检测任务。该实现包含了多项改进，提高了模型的检测性能和训练稳定性。

## 主要特点

- 基于 YOLOv8 架构的改进实现
- 优化的损失函数计算
- 完整的训练和验证流程
- 支持梯度累积和混合精度训练
- 灵活的配置系统
- 改进的数据增强策略

## 安装要求

- Python 3.8+
- PyTorch 1.8+
- Ultralytics
- torchvision
- numpy
- PyYAML
- tqdm
- logging

## 快速开始

1. 克隆仓库：
```bash
git clone [repository-url]
cd road_damage_detector
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据集：
   - 将数据集组织为 YOLO 格式
   - 创建 data.yaml 配置文件
   - 确保数据集路径正确配置

4. 开始训练：
```bash
python improved_yolov8/train.py --config improved_yolov8/config.yaml
```

## 配置说明

配置文件 (config.yaml) 包含以下主要部分：

```yaml
model:
  input_size: 640
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  steps: [100, 150]
  gamma: 0.1

training:
  epochs: 200
  batch_size: 16
  workers: 8
  device: 0
  freeze_backbone: false
  unfreeze_epoch: 1
  grad_accumulation: 1
  use_compile: false
  memory_fraction: 0.8

data:
  path: ./data
```

## 主要改进

1. 损失函数优化：
   - 改进的 CIoU 损失计算
   - 优化的坐标系转换
   - 更稳定的损失累积

2. 训练流程改进：
   - 支持梯度累积
   - 自适应的学习率调整
   - 改进的 backbone 冻结策略

3. 验证过程优化：
   - 更准确的 mAP 计算
   - 优化的 NMS 处理
   - 完整的评估指标

## 模型导出

训练完成后，模型会自动导出为 ONNX 格式，支持在不同平台上部署：

```bash
# 模型将保存在 runs/train/exp*/model.onnx
```

## 注意事项

1. 确保数据集格式正确，遵循 YOLO 格式要求
2. 根据显存大小调整 batch_size 和 memory_fraction
3. 对于大型数据集，建议启用梯度累积
4. 使用 `freeze_backbone` 可以加快训练速度

## 常见问题

1. 内存不足：
   - 减小 batch_size
   - 启用梯度累积
   - 调整 memory_fraction

2. 训练不稳定：
   - 检查学习率设置
   - 确认数据预处理正确
   - 尝试启用 backbone 冻结

## 许可证

[添加许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 引用

如果您使用了这个项目，请引用：

```
[添加引用信息]
``` 