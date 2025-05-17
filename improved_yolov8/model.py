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
import math
from types import SimpleNamespace

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

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

class C2f_Faster_EMA(C2f):
    """C2f块加入EMA注意力机制
    
    在标准C2f块基础上添加EMA注意力机制，提高特征提取能力
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 确保self.c的值正确设置
        self.c = int(c2 * e)  # 中间通道数
        self.ema = EMAAttention(self.c)
        logger.info(f"初始化C2f_Faster_EMA: 输入通道={c1}, 输出通道={c2}, 中间通道={self.c}")
        
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
        
        # 应用最终的卷积层 cv2，将通道数调整为 c2
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
        repeat (int): DyHead重复次数，默认2次
        use_dfl (bool): 是否使用DFL，默认True
    """
    def __init__(self, c1, c2, num_classes=80, anchors=None, repeat=2, use_dfl=True):
        super().__init__()
        self.nc = num_classes  # 类别数
        self.reg_max = 15  # 分类器的最大回归值，与v8DetectionLoss对齐
        # self.use_dfl = use_dfl # use_dfl 不再用于决定 self.no 的计算

        # 统一输出通道数计算，与 v8DetectionLoss 对齐
        # DFL模式: nc + reg_max * 4 (xywh)
        self.no = self.nc + 4 * self.reg_max          # 6 + 4*15 = 66

        self.nl = len(c1) if isinstance(c1, list) else 3  # 检测层数
        self.na = len(anchors[0]) // 2 if anchors else 1  # 每层锚点数
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) if anchors else None
        self.anchor_free = anchors is None  # 标记是否为无锚点模式
        self.repeat = repeat  # DyHead重复次数
        
        # YOLO风格的偏置初始化
        pi = 0.01  # 初始目标概率
        self._bias = -math.log((1 - pi) / pi)  # 使用YOLO的偏置初始化公式
        
        # 添加Ultralytics YOLO检测头所需的属性
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # 特征图步长
        self.bbox_format = 'xywh'  # 边界框格式
        self.nkpt = 0  # 关键点数量（如果有）
        self.device = None  # 将在模型初始化后设置
        
        # args字典，包含超参数
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
        
        # 为每个重复级别创建卷积层
        self.conv_layers = nn.ModuleList()
        for r in range(self.repeat):
            # 每个重复级别有nl个卷积层
            conv_layer = nn.ModuleList()
            for i in range(self.nl):
                channels = c1_list[i]
                # 如果不是最后一个重复层，输出通道应与输入相同
                out_channels = self.no if r == self.repeat - 1 else channels
                
                # 创建卷积
                conv = nn.Conv2d(channels, out_channels, 1)
                
                # 初始化卷积权重和偏置
                if r == self.repeat - 1:  # 只为最后一层进行特殊初始化
                    conv.bias.data.fill_(self._bias)  # 使用计算好的偏置
                    conv.weight.data.fill_(0.01)  # 权重仍使用小值初始化
                    
                conv_layer.append(conv)
            self.conv_layers.append(conv_layer)
        
        # 更新日志信息
        logger.info(f"初始化DyHead: 输入通道={c1}, 类别数={num_classes}, reg_max={self.reg_max}, 输出通道={self.no} (公式: nc + 4*reg_max)")
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (list): 特征图列表 [P3, P4, P5]，通道数分别为 [c1[0], c1[1], c1[2]]
            
        返回:
            list: 检测输出列表，标准YOLOv8检测头格式 [bs, no, ny, nx]
        """
        z = []  # 推理输出
        features = list(x)  # 复制输入特征，以便在迭代中修改
        
        # 对每个检测层进行处理
        for i in range(self.nl):
            feat = features[i]  # 获取特征图
            
            # 应用尺度注意力
            feat = self.scale_att[i](feat)
            
            # 对每个重复级别应用卷积
            for r in range(self.repeat):
                if r < self.repeat - 1:
                    # 中间层的输出反馈到特征图
                    feat = self.conv_layers[r][i](feat)
                    features[i] = feat
                else:
                    # 最后一层的输出用于检测
                    final_feat = self.conv_layers[r][i](feat)
                    
                    # 重塑为YOLOv8标准输出格式 [bs, no, ny, nx]
                    bs, _, ny, nx = final_feat.shape
                    final_feat = final_feat.view(bs, self.no, ny, nx)
                    z.append(final_feat)
                    
                    # 记录输出形状
                    logger.debug(f"DyHead输出层{i}形状: {final_feat.shape}, 通道数={self.no}")
        
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
        self.p4_adjust_conv = Conv(self.ch_p4_cat, self.ch_p4, 1, 1)  # 384→256，保持与DyHead输入一致
        
        # 检测头 - 使用DyHead，重复2次
        self.detect = DyHead(
            c1=[self.ch_p3, self.ch_p4, self.ch_p5],  # [128, 256, 512]
            c2=self.num_classes,
            num_classes=self.num_classes,
            repeat=2,  # 重复DyHead 2次，按论文最佳点
            use_dfl=True  # 使用DFL
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
        args['reg_max'] = 15  # 分类器的最大回归值，兼容DFL
            
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