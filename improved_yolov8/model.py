"""
改进的YOLOv8s模型 - 路面损伤检测
基于YOLOv8s模型架构，添加EMA注意力机制和动态检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, Conv, SPPF
from ultralytics.utils.tal import make_anchors
from ultralytics.data.loaders import LoadImagesAndVideos
import logging
import yaml
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class EMAAttention(nn.Module):
    """指数移动平均(EMA)注意力机制
    
    使用一维卷积和指数移动平均来捕获长距离依赖关系
    """
    def __init__(self, c, kernel_size=3, alpha=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=kernel_size, padding=kernel_size//2, groups=c)
        self.alpha = alpha
        
    def forward(self, x):
        # 原始输入形状: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 重塑为 [B, C, H*W]
        x_flat = x.reshape(B, C, -1)
        
        # 应用一维深度卷积
        conv_out = self.conv(x_flat)
        
        # 指数移动平均
        ema_out = x_flat.clone()
        for i in range(1, x_flat.shape[2]):
            ema_out[:, :, i] = self.alpha * x_flat[:, :, i] + (1 - self.alpha) * ema_out[:, :, i-1]
        
        # 融合并重塑回原始形状
        out = (conv_out + ema_out).reshape(B, C, H, W)
        return out + x  # 残差连接

class C2f_Faster_EMA(C2f):
    """C2f块加入EMA注意力机制
    
    在标准C2f块基础上添加EMA注意力机制，提高特征提取能力
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.ema = EMAAttention(self.c) # 使用 C2f 定义的 self.c 初始化
        
    def forward(self, x):
        # 标准 C2f 的前向传播逻辑
        y = list(self.cv1(x).chunk(2, 1))  # y[0], y[1] 的通道数都是 self.c

        # 应用 n 个 Bottleneck 模块，并在每个模块后应用 EMAAttention
        for m in self.m:
            bottleneck_out = m(y[-1])    # Bottleneck 输出通道数为 self.c
            ema_out = self.ema(bottleneck_out) # 应用 EMA，输入/输出通道数都为 self.c
            y.append(ema_out) # 添加经过 EMA 处理的特征

        # 拼接所有特征 (y[0], y[1], 和 n 个 ema_out)
        # 总通道数为 (2 + n) * self.c
        concatenated_features = torch.cat(y, 1)

        # 应用最终的卷积层 cv2
        return self.cv2(concatenated_features)

class SimSPPF(SPPF):
    """简化的空间金字塔池化模块
    
    使用ReLU替换SiLU以提高计算效率
    """
    def __init__(self, c1, c2, k=5):
        super().__init__(c1, c2, k)
        # 将激活函数改为ReLU提高计算效率
        self.cv1.act = nn.ReLU()
        self.cv2.act = nn.ReLU()

class ScaleAttention(nn.Module):
    """尺度注意力机制
    
    针对不同尺度的特征进行自适应加权
    
    参数:
        channels (int): 输入通道数
    """
    def __init__(self, channels):
        super().__init__()
        # 处理通道数
        if isinstance(channels, (list, tuple)):
            logger.warning(f"ScaleAttention收到通道数列表{channels}，使用第一个元素{channels[0]}")
            channels = channels[0]
        
        mid_channels = max(channels // 4, 1)  # 确保中间层通道数至少为1
        
        # 注意力模块
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        logger.info(f"初始化ScaleAttention: 输入通道={channels}, 中间通道={mid_channels}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入特征图 [B, C, H, W]
            
        返回:
            tensor: 加权后的特征图 [B, C, H, W]
        """
        att = self.fc(x)
        return x * att

class DyHead(nn.Module):
    """动态检测头
    
    具有尺度、空间和任务感知注意力的动态检测头
    
    参数:
        c1 (int or list): 输入通道数，可以是整数（所有层相同）或列表（每层不同）
        c2 (int): 输出通道数，通常等于类别数
        num_classes (int): 类别数
        anchors (list, optional): 锚点配置，如果为None则不使用锚点（anchor-free）
    """
    def __init__(self, c1, c2, num_classes=80, anchors=None):
        super().__init__()
        self.nc = num_classes  # 类别数
        self.no = num_classes + 5  # 输出数 (类别 + x,y,w,h,obj) - 确保与自定义损失函数兼容
        self.nl = len(c1) if isinstance(c1, list) else 3  # 检测层数
        self.na = len(anchors[0]) // 2 if anchors else 1  # 每层锚点数，如果没有提供锚点则设为1
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) if anchors else None
        self.anchor_free = anchors is None  # 标记是否为无锚点模式
        
        # 添加Ultralytics YOLO检测头所需的属性
        # 这些属性是v8DetectionLoss所必需的
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # 特征图步长
        self.reg_max = 16  # 分类器的最大回归值，用于DFL（Distribution Focal Loss）
        
        # 以下属性可能也会被v8DetectionLoss访问
        self.use_dfl = False  # 简化：设置为False以避免DFL复杂性
        self.bbox_format = 'xywh'  # 边界框格式
        self.nkpt = 0  # 关键点数量（如果有）
        self.device = None  # 将在模型初始化后设置
        
        # args字典，包含超参数，一些ultralytics组件可能会查找
        self.args = {
            'cls': 0.5,  # 分类损失权重
            'box': 7.5,  # 边界框损失权重
            'dfl': 1.5,  # DFL损失权重
            'reg_max': self.reg_max,  # 分类器的最大回归值
            'stride': self.stride.tolist(),  # 步长列表
            'kobj': 1.0,  # 关键点目标性损失权重（如果有）
            'fl_gamma': 0.0,  # 焦点损失gamma值
            'anchor_t': 4.0  # 锚点阈值
        }
        
        # 尺度注意力 - 为每个检测层创建独立的注意力模块
        self.scale_att = nn.ModuleList()
        c1_list = c1 if isinstance(c1, list) else [c1] * self.nl
        
        for i in range(self.nl):
            channels = c1_list[i]
            self.scale_att.append(ScaleAttention(channels))
        
        # 动态检测头卷积 - 每层一个卷积
        self.cv = nn.ModuleList()
        for i in range(self.nl):
            channels = c1_list[i]
            # 我们使用统一的输出通道: (nc + 5) * na
            # 这与CustomDetectionLoss的期望格式相匹配：
            # - 前4个通道用于边界框预测（x,y,w,h）
            # - 第5个通道用于目标性预测（objectness）
            # - 剩余nc个通道用于类别预测
            out_channels = self.no * self.na  # no = nc + 5
            self.cv.append(nn.Conv2d(channels, out_channels, 1))
            
            # 初始化卷积权重
            self.cv[-1].bias.data.fill_(0.0)
            self.cv[-1].weight.data.fill_(0.01)
        
        # Dropout正则化
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (list): 特征图列表 [P3, P4, P5]，通道数分别为 [c1[0], c1[1], c1[2]]
            
        返回:
            list: 检测输出列表，标准YOLOv8检测头格式 [bs, na, ny, nx, no]
        """
        z = []  # 推理输出
        
        for i in range(self.nl):
            # 应用尺度注意力
            feat = x[i]  # 特征图
            feat = self.scale_att[i](feat)
            
            # 应用dropout正则化
            feat = self.dropout(feat)
            
            # 应用检测卷积
            feat = self.cv[i](feat)
            
            # 重塑输出 [bs, na*no, ny, nx] → [bs, na, ny, nx, no]
            bs, _, ny, nx = feat.shape
            # 重塑为 YOLO标准输出格式 [bs, na, ny, nx, no]
            feat = feat.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 添加到输出列表
            z.append(feat)
            
        return z

class ImprovedYoloV8s(nn.Module):
    """改进的YOLOv8s模型
    
    特点:
    1. 使用C2f-Faster-EMA作为骨干网络，提高特征提取能力
    2. SimSPPF加速特征融合
    3. DyHead动态检测头提高检测准确性
    
    结构与Ultralytics YOLO兼容
    """
    def __init__(self, config_path=None):
        super().__init__()
        
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"成功加载配置: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
        
        # 模型配置
        model_config = self.config['model']
        classes_config = self.config['classes']
        
        self.num_classes = len(classes_config['names'])
        self.input_size = model_config['input_size']
        self.names = classes_config['names']  # 类别名称列表，兼容Ultralytics
        
        # 基础参数
        base_channels = 32  # 基础通道数
        
        # 模块列表构建（与Ultralytics YOLO兼容）
        self.backbone = nn.ModuleList()
        self.neck = nn.ModuleList()
        
        # 主干网络 - 增强版YOLOv8s
        # 添加到backbone
        self.backbone.append(Conv(3, base_channels, 3, 2))  # 输入层 (stem)
        
        # 下采样和特征提取阶段
        self.backbone.append(nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f_Faster_EMA(base_channels * 2, base_channels * 2, 2)
        ))  # dark2
        
        self.backbone.append(nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f_Faster_EMA(base_channels * 4, base_channels * 4, 2)
        ))  # dark3
        
        self.backbone.append(nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f_Faster_EMA(base_channels * 8, base_channels * 8, 2)
        ))  # dark4
        
        self.backbone.append(nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C2f_Faster_EMA(base_channels * 16, base_channels * 16, 2),
            SimSPPF(base_channels * 16, base_channels * 16)
        ))  # dark5
        
        # FPN-PAN 特征融合网络 (添加到neck)
        # FPN (自顶向下路径)
        self.neck.append(nn.Sequential(
            Conv(base_channels * 16, base_channels * 8, 1, 1),
            nn.Upsample(scale_factor=2)
        ))  # fpn1
        
        self.neck.append(nn.Sequential(
            C2f_Faster_EMA(base_channels * 16, base_channels * 8, 2, False),
            Conv(base_channels * 8, base_channels * 4, 1, 1),
            nn.Upsample(scale_factor=2)
        ))  # fpn2
        
        self.neck.append(C2f_Faster_EMA(base_channels * 8, base_channels * 4, 2, False))  # fpn3
        
        # PAN (自底向上路径)
        self.neck.append(nn.Sequential(
            Conv(base_channels * 4, base_channels * 4, 3, 2)  # p3(4b) -> down(4b)
        ))  # pan1
        
        self.neck.append(nn.Sequential(
            C2f_Faster_EMA(base_channels * 12, base_channels * 8, 2, False),  # Input: cat(pan1(4b=128), x4(8b=256)) = 12b=384. Output: 8b=256.
            Conv(base_channels * 8, base_channels * 8, 3, 2)  # Output: 8b=256
        ))  # pan2
        
        self.neck.append(C2f_Faster_EMA(base_channels * 24, base_channels * 16, 2, False))  # pan3
        
        # 检测头
        self.detect = DyHead(
            c1=[base_channels * 4, base_channels * 12, base_channels * 16],
            c2=self.num_classes,
            num_classes=self.num_classes
        )
        
        # 创建model属性，包含所有主要模块，与ultralytics兼容
        # 模型结构：backbone + neck + detect
        self.model = nn.ModuleList()
        self.model.extend(self.backbone)  # 添加骨干网络模块
        self.model.extend(self.neck)      # 添加颈部模块
        self.model.append(self.detect)    # 添加检测头作为model的最后一个元素
        
        # 设置stride和anchors，兼容DetectionLoss
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.anchors = None  # 无锚点模式
        
        # 设置超参数为args，兼容ultralytics
        args = {}
        # 合并training和loss配置
        if 'training' in self.config:
            args.update(self.config['training'])
        if 'loss' in self.config:
            args.update(self.config['loss'])
        
        # 确保有必要的损失权重
        if 'box' not in args:
            args['box'] = 7.5  # 默认box loss权重
        if 'cls' not in args:
            args['cls'] = 0.5  # 默认cls loss权重
        if 'dfl' not in args:
            args['dfl'] = 1.5  # 默认dfl loss权重
            
        # 设置DFL相关参数
        args['reg_max'] = 16  # 分类器的最大回归值，兼容DFL
            
        self.args = args
        
        # 确保检测头的stride与模型的一致
        self.detect.stride = self.stride
        
        # 确保检测头的reg_max和其他属性与模型一致
        self.detect.reg_max = args.get('reg_max', 16)
        
        # 更新检测头的args
        if hasattr(self.detect, 'args'):
            # 将模型参数合并到检测头参数
            for k, v in args.items():
                if isinstance(self.detect.args, dict):
                    self.detect.args[k] = v
                else:
                    setattr(self.detect.args, k, v)
        
        logger.info(f"初始化改进的YOLOv8s模型，类别数: {self.num_classes}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入图像张量，形状为 [B, 3, H, W]
            
        返回:
            Tensor or List[Tensor]: 检测结果
        """
        # 中间特征存储
        features = []
        
        # 主干网络前向传播 (backbone)
        x = self.backbone[0](x)  # stem
        x2 = self.backbone[1](x)  # dark2
        x3 = self.backbone[2](x2)  # dark3
        features.append(x3)  # 保存特征图P3
        x4 = self.backbone[3](x3)  # dark4
        features.append(x4)  # 保存特征图P4
        x5 = self.backbone[4](x4)  # dark5
        features.append(x5)  # 保存特征图P5
        
        # FPN - 自顶向下路径
        p5 = x5
        p4 = self.neck[0](p5)  # fpn1
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.neck[1](p4)  # fpn2
        p3 = torch.cat([p4, x3], dim=1)
        p3 = self.neck[2](p3)  # fpn3
        
        # PAN - 自底向上路径
        p4 = self.neck[3](p3)  # pan1
        p4 = torch.cat([p4, x4], dim=1)
        p5 = self.neck[4](p4)  # pan2
        p5 = torch.cat([p5, x5], dim=1)
        p5 = self.neck[5](p5)  # pan3
        
        # 检测输出
        return self.detect([p3, p4, p5])  # 返回检测结果列表
        
    def _initialize_biases(self):
        """初始化检测头的偏置，提高训练初期的稳定性
        
        这是Ultralytics YOLO中常见的做法
        """
        # 如果检测头支持偏置初始化，调用它
        if hasattr(self.detect, '_initialize_biases'):
            self.detect._initialize_biases()

    def to(self, device):
        """将模型移动到设备，并更新detect.device属性
        
        参数:
            device: 目标设备
            
        返回:
            model: 移动后的模型
        """
        model = super().to(device)
        if hasattr(model.detect, 'device'):
            model.detect.device = device
        return model 