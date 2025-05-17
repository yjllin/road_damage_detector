"""
改进的YOLOv8s模型包
"""

from .model import ImprovedYoloV8s
from .train import train, CustomDetectionLoss

__all__ = ['ImprovedYoloV8s', 'train', 'CustomDetectionLoss']