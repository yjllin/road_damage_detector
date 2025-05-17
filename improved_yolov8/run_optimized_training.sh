#!/usr/bin/env bash
# 使用优化设置的YOLOv8优化训练脚本

# 确保脚本在出错时停止执行
set -e

# 设置输出目录名称
RUN_NAME="optimized_yolov8"

# 打印系统信息
echo "系统信息:"
echo "- PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "- CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "- GPU型号: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# 打印优化信息
echo "启用以下优化:"
echo "- 向量化EMAAttention"
echo "- PyTorch 2.0 compile (如果可用)"
echo "- 向量化损失函数"
echo "- 优化的日志记录"
echo "- 梯度累积（提高大批次效率）"
echo "- 参数分组优化"
echo "- 恢复模型宽度为32通道"
echo "- DyHead重复2次"
echo "- 提高学习率到0.01"

# 启动训练，使用所有优化选项和更新的超参数
python improved_yolov8/train.py --data ./database/China_MotorBike --batch-size 16 --workers 8 --grad-accumulation 4 --lr 0.01 --unfreeze-epoch 1 --amp --use-compile --memory-fraction 0.8 --name ${RUN_NAME} --log-interval 100 --epochs 200

echo "优化训练完成！模型保存在 runs/train/${RUN_NAME}/" 