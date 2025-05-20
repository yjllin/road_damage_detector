"""
改进的YOLOv8s模型 - 路面损伤检测
基于YOLOv8s模型架构，添加EMA注意力机制和动态检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, Conv, SPPF, Detect
from ultralytics.utils.tal import make_anchors
from ultralytics.data.loaders import LoadImagesAndVideos
import logging
import yaml
from pathlib import Path
import math
from types import SimpleNamespace

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class DyHeadBlock(nn.Module):
    """动态头部块 - 改进版DyHead设计
    
    特点:
    1. 使用深度可分离卷积降低计算量
    2. 添加通道注意力机制
    3. 优化特征融合方式
    
    参数:
        c (int): 输入/输出通道数
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        
        # 空间自适应 - 使用深度可分离卷积
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c),  # 深度可分离卷积
            nn.BatchNorm2d(c),  # 添加BN层提高稳定性
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力 - 使用SE模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//4, c, 1),
            nn.Sigmoid()
        )
        
        # 尺度自适应 - 多尺度特征融合
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//4, c, 1),
            nn.Sigmoid()
        )
        
        # 任务自适应 - 特征增强
        self.task_conv = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(c*2, c, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        logger.info(f"初始化改进版DyHeadBlock: 通道数={c}")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor或list): 输入特征图或特征图列表
            
        返回:
            list或tensor: 处理后的特征
        """
        if isinstance(x, list):
            return [self.forward_single(xi) for xi in x]
        else:
            return self.forward_single(x)
            
    def forward_single(self, x):
        """处理单个特征图
        
        参数:
            x (tensor): 输入特征图
            
        返回:
            tensor: 处理后的特征图
        """
        # 空间特征提取
        spatial_feat = self.spatial_conv(x)
        
        # 通道注意力
        channel_att = self.channel_attention(x)
        channel_feat = x * channel_att
        
        # 尺度注意力
        scale_att = self.scale_attention(x)
        scale_feat = x * scale_att
        
        # 特征融合
        fused = torch.cat([spatial_feat, channel_feat], dim=1)
        fused = self.fusion(fused)
        
        # 任务自适应处理
        out = self.task_conv(fused)
        
        return out + x  # 残差连接

class DyDetect(Detect):
    """动态检测头
    
    基于DyHead设计的检测头，增强了特征自适应能力
    
    参数:
        nc (int): 类别数
        ch (tuple): 输入通道列表
        reg_max (int): DFL分类器最大值
    """
    def __init__(self, nc, ch, reg_max=16):
        super().__init__(nc, ch)
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max  # 输出通道数 = nc + 4*reg_max
        
        # 为每个特征层级创建独立的DyHead处理模块
        self.dyhead_modules = nn.ModuleList()
        for c in ch:  # 对每个输入通道数分别创建处理模块
            dyhead = nn.Sequential(
                DyHeadBlock(c),  # 第一层DyHead
                DyHeadBlock(c)   # 第二层DyHead
            )
            self.dyhead_modules.append(dyhead)
            
        # 回归 & 分类独立卷积
        self.cv_reg = nn.ModuleList([nn.Conv2d(c, 4*reg_max, 1) for c in ch])
        self.cv_cls = nn.ModuleList([nn.Conv2d(c, nc, 1) for c in ch])
        
        logger.info(f"初始化DyDetect: nc={nc}, reg_max={reg_max}, no={self.no}")

    def _reorder(self, reg, cls):
        """重新排列回归和分类特征，确保通道顺序与验证函数期望的一致
        
        参数:
            reg (Tensor): 回归特征 [B, 4*reg_max, H, W]
            cls (Tensor): 分类特征 [B, nc, H, W]
            
        返回:
            Tensor: 按照 [B, no, H, W] 格式排列的特征图
        """
        return torch.cat([reg, cls], dim=1)  # 先回归后分类

    def forward(self, x):
        """前向传播
        
        参数:
            x (list): 多尺度特征图列表 [p3, p4, p5]
                
        返回:
            list: 多尺度检测结果列表, 每个元素形状为 [B, no, H, W]
        """
        # 对每个特征层级分别应用对应的DyHead处理
        outs = []
        for i, (feat, dyhead) in enumerate(zip(x, self.dyhead_modules)):
            # 应用DyHead处理
            feat = dyhead(feat)
            
            # 预测回归和分类
            reg = self.cv_reg[i](feat)       # [B,4*reg_max,H,W]
            cls = self.cv_cls[i](feat)       # [B,nc,H,W]
            outs.append(self._reorder(reg, cls))  # 使用_reorder确保顺序一致
            
        return outs

class EMAAttention(nn.Module):
    """指数移动平均(EMA)注意力机制
    
    使用一维卷积和指数移动平均来捕获长距离依赖关系
    使用向量化实现，大幅提高性能
    """
    def __init__(self, c, kernel_size=3, alpha=0.1, max_len=32768):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=kernel_size, padding=kernel_size//2, groups=c)
        self.alpha = alpha
        self.max_len = max_len
        
        # 预计算衰减因子并注册为buffer，使用float64提高精度
        positions = torch.arange(max_len, dtype=torch.float64)
        decay = ((1 - alpha) ** positions).to(torch.float32)
        self.register_buffer('decay', decay)
        logger.info(f"初始化EMAAttention: 通道数={c}, alpha={alpha}, 最大长度={max_len}")
        
    def _extend_decay(self, L):
        """动态扩展衰减因子，使用float64计算以保持精度"""
        if L <= self.max_len:
            return
        
        logger.info(f"扩展EMA衰减因子从{self.max_len}到{L}")
        self.max_len = L
        device = self.decay.device
        dtype = self.decay.dtype
        
        # 使用float64计算新的衰减因子
        positions = torch.arange(L, dtype=torch.float64, device=device)
        decay = (1 - self.alpha) ** positions
        
        # 转换为目标dtype并更新buffer
        self.register_buffer('decay', decay.to(dtype), persistent=False)
        
    def forward(self, x):
        # 原始输入形状: [B, C, H, W]
        B, C, H, W = x.shape
        L = H * W
        
        # 如果需要，扩展衰减因子
        if L > self.max_len:
            self._extend_decay(L)
        
        # 重塑为 [B, C, H*W]
        x_flat = x.flatten(2)  # B,C,L
        
        # 使用预计算的衰减因子,并截取所需长度
        decay = self.decay[:L].view(1, 1, -1)
        
        # 计算EMA
        weighted_input = x_flat * self.alpha * decay
        cumsum = torch.cumsum(weighted_input, dim=-1)
        cumsum_decay = torch.cumsum(self.alpha * decay, dim=-1)
        ema = cumsum / (cumsum_decay + 1e-7)
        
        # 融合并重塑回原始形状
        out = (ema + self.conv(x_flat)).view(B, C, H, W)
        return out + x  # 残差连接

class C2f_Faster_EMA(nn.Module):
    """改进的C2f模块，使用EMA注意力机制
    
    特点:
    1. 使用EMA注意力机制增强特征提取
    2. 优化计算效率
    3. 添加残差连接
    
    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 重复次数
        shortcut (bool): 是否使用shortcut连接
    """
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2 // 2  # 通道数减半
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, 1.0, k=((1, 1), (3, 3))) for _ in range(n))
        
        # EMA注意力机制
        self.ema = EMA(self.c)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入特征图
            
        返回:
            tensor: 处理后的特征图
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # 应用EMA注意力
        y = [self.ema(yi) for yi in y]
        
        return self.cv2(torch.cat(y, 1))

class EMA(nn.Module):
    """指数移动平均注意力机制
    
    特点:
    1. 自适应特征增强
    2. 降低计算复杂度
    3. 提高特征提取效率
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 特征融合
        self.fc = nn.Sequential(
            nn.Linear(c, c // 4),
            nn.ReLU(inplace=True),
            nn.Linear(c // 4, c),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入特征图
            
        返回:
            tensor: 增强后的特征图
        """
        # 平均池化和最大池化
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        
        # 特征融合
        out = self.fc(avg_out + max_out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        
        return x * out.expand_as(x)

class Bottleneck(nn.Module):
    """改进的Bottleneck模块
    
    特点:
    1. 使用深度可分离卷积
    2. 添加BN层
    3. 优化激活函数
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=((1, 1), (3, 3))):
        super().__init__()
        c_ = int(c2 * g)
        self.cv1 = Conv(c1, c_, int(k[0][0]), 1)  # 确保k[0]是整数
        self.cv2 = Conv(c_, c2, int(k[1][0]), 1, g=int(g))  # 确保k[1]和g都是整数
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入特征图
            
        返回:
            tensor: 处理后的特征图
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SimSPPF(nn.Module):
    """简化的空间金字塔池化模块
    
    特点:
    1. 使用ReLU替换SiLU提高计算效率
    2. 添加注意力机制
    3. 优化特征融合
    
    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 池化核大小
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_ * 4, c_ * 4 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_ * 4 // 16, c_ * 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (tensor): 输入特征图
            
        返回:
            tensor: 处理后的特征图
        """
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        # 特征拼接
        y = torch.cat((x, y1, y2, y3), 1)
        
        # 应用注意力
        att = self.attention(y)
        y = y * att
        
        return self.cv2(y)

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

class ImprovedYoloV8s(nn.Module):
    """改进的YOLOv8s模型
    
    特点:
    1. 使用C2f-Faster-EMA作为骨干网络，提高特征提取能力
    2. SimSPPF加速特征融合
    3. DyDetect动态检测头提高检测准确性
    
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
        
        # 基础参数 - 恢复与YOLOv8s一致的宽度
        base_channels = 32  # 与YOLOv8-s对齐，原来是16
        self.bc = base_channels  # 保存base_channels为实例变量
        
        # 计算各层通道数
        self.ch_p3 = self.bc * 4   # P3通道数 (128)
        self.ch_p4 = self.bc * 8   # P4通道数 (256)
        self.ch_p5 = self.bc * 16  # P5通道数 (512)
        self.ch_p4_cat = self.ch_p3 + self.ch_p4  # P4拼接后通道数 (384)
        
        # 模块列表构建（与Ultralytics YOLO兼容）
        self.backbone = nn.ModuleList()
        self.neck = nn.ModuleList()
        
        # 主干网络 - 增强版YOLOv8s
        # 添加到backbone
        self.backbone.append(Conv(3, self.bc, 3, 2))  # 输入层 (stem)
        
        # 下采样和特征提取阶段
        self.backbone.append(nn.Sequential(
            Conv(self.bc, self.bc * 2, 3, 2),
            C2f_Faster_EMA(self.bc * 2, self.bc * 2, 1)  # 保持n=1
        ))  # dark2
        
        self.backbone.append(nn.Sequential(
            Conv(self.bc * 2, self.ch_p3, 3, 2),
            C2f_Faster_EMA(self.ch_p3, self.ch_p3, 2)  # 增加重复块数到2
        ))  # dark3
        
        self.backbone.append(nn.Sequential(
            Conv(self.ch_p3, self.ch_p4, 3, 2),
            C2f_Faster_EMA(self.ch_p4, self.ch_p4, 2)  # 增加重复块数到2
        ))  # dark4
        
        self.backbone.append(nn.Sequential(
            Conv(self.ch_p4, self.ch_p5, 3, 2),
            C2f_Faster_EMA(self.ch_p5, self.ch_p5, 1),  # 保持n=1
            SimSPPF(self.ch_p5, self.ch_p5)
        ))  # dark5
        
        # FPN-PAN 特征融合网络 (添加到neck)
        # FPN (自顶向下路径)
        self.neck.append(nn.Sequential(
            Conv(self.ch_p5, self.ch_p4, 1, 1),
            nn.Upsample(scale_factor=2)
        ))  # fpn1
        
        self.neck.append(nn.Sequential(
            C2f_Faster_EMA(self.ch_p4 * 2, self.ch_p4, 2, False),
            Conv(self.ch_p4, self.ch_p3, 1, 1),
            nn.Upsample(scale_factor=2)
        ))  # fpn2
        
        self.neck.append(C2f_Faster_EMA(self.ch_p3 * 2, self.ch_p3, 2, False))  # fpn3
        
        # PAN (自底向上路径)
        self.neck.append(nn.Sequential(
            Conv(self.ch_p3, self.ch_p3, 3, 2)  # p3(128) -> down(128)
        ))  # pan1
        
        # 修改pan2的通道数计算，确保通道数匹配
        self.neck.append(nn.Sequential(
            # 先调整p4_adj的通道数
            Conv(self.ch_p4, self.ch_p4, 1, 1),  # 保持256通道
            C2f_Faster_EMA(self.ch_p4, self.ch_p4, 2, False),  # Input: 256 -> 256
            Conv(self.ch_p4, self.ch_p4, 3, 2)  # 256 -> 256，下采样
        ))  # pan2
        
        # 修改pan3的通道数计算
        self.neck.append(C2f_Faster_EMA(self.ch_p4 + self.ch_p5, self.ch_p5, 2, False))  # Input: cat(256+512) = 768 -> 512
        
        # 修改p4调整卷积层的通道数
        self.p4_adjust_conv = Conv(self.ch_p4_cat, self.ch_p4, 1, 1)  # 384→256，保持与DyDetect输入一致
        
        # 检测头 - 使用DyDetect，重复2次
        self.detect = DyDetect(
            nc=self.num_classes,
            ch=(self.ch_p3, self.ch_p4, self.ch_p5)
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
        self.hyp = SimpleNamespace(**self.args)
        
        # 确保检测头的stride与模型的一致
        self.detect.stride = self.stride
        
        # 确保检测头的reg_max和其他属性与模型一致
        self.detect.reg_max = args['reg_max']  # 同步reg_max值
        
        # 更新检测头的args
        if hasattr(self.detect, 'args'):
            # 将模型参数合并到检测头参数
            for k, v in args.items():
                if isinstance(self.detect.args, dict):
                    self.detect.args[k] = v
                else:
                    setattr(self.detect.args, k, v)
        
        logger.info(f"初始化改进的YOLOv8s模型，类别数: {self.num_classes}")
        
        # 添加编译标志
        self.is_compiled = False
        
    def compile_model(self):
        """编译模型，确保只编译一次"""
        if not self.is_compiled and hasattr(torch, 'compile'):
            try:
                logger.info("编译模型...")
                compiled = torch.compile(self, mode="reduce-overhead")
                compiled.is_compiled = True
                logger.info("模型编译成功")
                return compiled
            except Exception as e:
                logger.error(f"模型编译失败: {e}")
        return self
        
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
        x3 = self.backbone[2](x2)  # dark3 (128)
        features.append(x3)  # 保存特征图P3
        x4 = self.backbone[3](x3)  # dark4 (256)
        features.append(x4)  # 保存特征图P4
        x5 = self.backbone[4](x4)  # dark5 (512)
        features.append(x5)  # 保存特征图P5
        
        # FPN - 自顶向下路径
        p5 = x5  # 512
        p4_up = self.neck[0](p5)  # fpn1: 512->256
        p4_cat = torch.cat([p4_up, x4], dim=1)  # 256+256=512
        p4 = self.neck[1](p4_cat)  # fpn2: 512->256->128
        p3_cat = torch.cat([p4, x3], dim=1)  # 128+128=256
        p3 = self.neck[2](p3_cat)  # fpn3: 256->128
        
        # PAN - 自底向上路径
        p3_down = self.neck[3](p3)  # pan1: 128->128
        
        # 调整p4的通道数
        p4_cat_2 = torch.cat([p3_down, x4], dim=1)  # 128+256=384
        p4_adj = self.p4_adjust_conv(p4_cat_2)  # 384->256
        
        # pan2处理
        p4_down = self.neck[4][0](p4_adj)  # 保持256通道
        p4_down = self.neck[4][1](p4_down)  # C2f_Faster_EMA: 256->256
        p4_down = self.neck[4][2](p4_down)  # 下采样: 256->256
        
        # pan3处理
        p5_cat = torch.cat([p4_down, x5], dim=1)  # 256+512=768
        p5_out = self.neck[5](p5_cat)  # 768->512
        
        # 检测输出
        return self.detect([p3, p4_adj, p5_out])  # [128, 256, 512]
        
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