#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化训练数据集，用于调试标注问题
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics.data.dataset import YOLODataset
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from collections import Counter, defaultdict
import matplotlib.font_manager as fm

# 配置中文字体支持
def setup_chinese_font():
    """
    配置matplotlib支持中文显示
    """
    # 尝试查找系统中的中文字体
    chinese_fonts = [
        # Windows字体
        'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi',
        # Linux字体
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'AR PL New Sung', 
        # macOS字体
        'Hiragino Sans GB', 'STHeiti', 'STSong', 'STFangsong', 'PingFang SC',
    ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font in chinese_fonts:
        try:
            if any(font.lower() in f.lower() for f in fm.findSystemFonts()):
                available_fonts.append(font)
                print(f"找到可用中文字体: {font}")
        except Exception:
            pass
    
    # 如果找到合适的字体，设置matplotlib使用第一个可用的中文字体
    if available_fonts:
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        print(f"设置matplotlib使用字体: {available_fonts[0]}")
    else:
        print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")
        # 使用matplotlib默认的回退方案，尝试使用Unicode回退字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.fallback'] = True
        
    # 在某些情况下，可能需要调整下面的设置
    plt.rcParams['font.family'] = 'sans-serif'

def load_font():
    """加载中文字体"""
    try:
        # 尝试加载系统中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
            '/System/Library/Fonts/PingFang.ttc'  # macOS
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 20)
        return ImageFont.load_default()
    except Exception as e:
        print(f"加载字体失败: {e}")
        return ImageFont.load_default()

def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    """
    在图像上绘制一个边界框
    
    参数:
        box: 边界框坐标 [x1, y1, x2, y2]
        img: PIL Image对象
        color: RGB颜色元组
        label: 标签文本
        line_thickness: 线条粗细
    """
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    
    # 转换为整数坐标
    box = [int(x) for x in box]
    
    # 绘制矩形
    for i in range(line_thickness):
        draw.rectangle([box[0] + i, box[1] + i, box[2] - i, box[3] - i],
                      outline=color)
    
    # 如果有标签，绘制标签背景和文本
    if label:
        font = load_font()
        
        # 使用新的方法获取文本尺寸（兼容新版本PIL）
        try:
            # 尝试使用textbbox (PIL 9.2.0+)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = right - left, bottom - top
        except AttributeError:
            try:
                # 尝试使用getbbox (PIL 8.0.0+)
                bbox = font.getbbox(label)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # 最后尝试使用getsize (已废弃但某些版本可能存在)
                text_w, text_h = font.getsize(label)
        
        outside = box[1] - text_h - 3 >= 0  # 标签是否能放在框上方
        
        # 绘制标签背景
        if outside:
            draw.rectangle([box[0], box[1] - text_h - 3,
                          box[0] + text_w + 3, box[1]],
                          fill=color)
        else:
            draw.rectangle([box[0], box[1],
                          box[0] + text_w + 3, box[1] + text_h + 3],
                          fill=color)
        
        # 绘制文本
        text_color = (255, 255, 255)  # 白色文本
        if outside:
            draw.text((box[0] + 2, box[1] - text_h - 2),
                     label, fill=text_color, font=font)
        else:
            draw.text((box[0] + 2, box[1] + 2),
                     label, fill=text_color, font=font)

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    参数:
        box1, box2: [x1, y1, x2, y2] 格式的边界框
    """
    # 计算交集区域的坐标
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # 计算交集区域的面积
    if x2_min < x1_max or y2_min < y1_max:
        return 0  # 没有交集
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算IoU
    iou = intersection / (box1_area + box2_area - intersection)
    
    return iou

def analyze_bbox_dimensions(dataset, class_names, save_dir):
    """
    分析边界框的尺寸分布
    
    参数:
        dataset: YOLODataset实例
        class_names: 类别名称列表
        save_dir: 保存分析结果的目录
    """
    print("\n== 分析边界框尺寸 ==")
    
    # 存储所有边界框的信息
    width_list = []
    height_list = []
    ratio_list = []
    area_list = []
    class_list = []
    
    # 遍历数据集
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"处理样本 {idx}/{len(dataset)}")
            
        sample = dataset[idx]
        bboxes = sample['bboxes']  # 归一化的 xywh 格式
        cls = sample['cls']
        img = sample['img']
        
        # 获取图像尺寸
        img_height, img_width = img.shape[1:3]
        
        # 处理每个边界框
        for box, cls_idx in zip(bboxes, cls):
            x, y, w, h = box
            
            # 转换为绝对像素大小
            width_px = w * img_width
            height_px = h * img_height
            area_px = width_px * height_px
            ratio = width_px / height_px if height_px > 0 else 0
            
            # 记录数据
            width_list.append(width_px)
            height_list.append(height_px)
            ratio_list.append(ratio)
            area_list.append(area_px)
            class_list.append(int(cls_idx))
    
    # 创建DataFrame以便分析
    df = pd.DataFrame({
        'width': width_list,
        'height': height_list,
        'ratio': ratio_list,
        'area': area_list,
        'class': class_list
    })
    
    # 保存原始数据
    df.to_csv(save_dir / 'bbox_dimensions.csv', index=False)
    
    # 分析小框的情况
    small_bbox_count = len(df[(df['width'] < 3) | (df['height'] < 3)])
    extreme_ratio_count = len(df[(df['ratio'] < 0.05) | (df['ratio'] > 20)])
    
    print(f"总框数: {len(df)}")
    print(f"小于3像素的框数量: {small_bbox_count} ({small_bbox_count/len(df)*100:.2f}%)")
    print(f"宽高比异常的框数量: {extreme_ratio_count} ({extreme_ratio_count/len(df)*100:.2f}%)")
    
    # 按类别统计 - 修改聚合方式，分步计算而不是使用复杂的字典
    class_stats = []
    for cls_id, group in df.groupby('class'):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        stats = {
            'class': cls_id,
            'class_name': cls_name,
            'count': len(group),
            'width_mean': group['width'].mean(),
            'width_min': group['width'].min(),
            'width_max': group['width'].max(),
            'height_mean': group['height'].mean(),
            'height_min': group['height'].min(),
            'height_max': group['height'].max(),
            'ratio_mean': group['ratio'].mean(),
            'ratio_min': group['ratio'].min(),
            'ratio_max': group['ratio'].max(),
            'area_mean': group['area'].mean(),
            'area_min': group['area'].min(),
            'area_max': group['area'].max()
        }
        class_stats.append(stats)
    
    # 转换为DataFrame
    class_stats_df = pd.DataFrame(class_stats)
    
    # 保存类别统计
    class_stats_df.to_csv(save_dir / 'bbox_stats_by_class.csv', index=False)
    
    # 可视化
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 安全绘图函数，避免KDE错误
        def safe_histplot(data, ax, title, vertical_line=None, vline_label=None):
            # 检查数据是否有足够的不同值来绘制KDE
            unique_values = data.nunique()
            use_kde = unique_values > 5  # 至少需要5个不同的值
            
            if unique_values <= 1:
                # 如果只有一个值，使用条形图
                ax.bar([data.iloc[0]], [len(data)], width=0.1)
                ax.set_title(f"{title} (单一值: {data.iloc[0]:.2f})")
            else:
                # 使用直方图，只在数据足够丰富时启用KDE
                sns.histplot(data, bins=min(50, unique_values), ax=ax, kde=use_kde)
                ax.set_title(title)
            
            # 添加垂直线
            if vertical_line is not None:
                ax.axvline(x=vertical_line, color='r', linestyle='--', label=vline_label)
                ax.legend()
        
        # 宽度分布
        width_data = df['width'].clip(0, 100)
        safe_histplot(width_data, axes[0, 0], '边界框宽度分布 (像素)', 3, '3px')
        
        # 高度分布
        height_data = df['height'].clip(0, 100)
        safe_histplot(height_data, axes[0, 1], '边界框高度分布 (像素)', 3, '3px')
        
        # 宽高比分布
        ratio_data = df['ratio'].clip(0, 5)
        if ratio_data.nunique() > 1:
            safe_histplot(ratio_data, axes[1, 0], '宽高比分布 (width/height)', 0.05, 'ratio=0.05')
            # 添加第二条线
            if ratio_data.max() > 5:
                axes[1, 0].axvline(x=5, color='orange', linestyle='--', label='ratio=5')
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, f"所有比例值都相同: {ratio_data.iloc[0]:.2f}", 
                          horizontalalignment='center', verticalalignment='center')
            axes[1, 0].set_title('宽高比分布 (width/height)')
        
        # 面积分布
        area_data = np.sqrt(df['area'].clip(0, 10000))
        safe_histplot(area_data, axes[1, 1], '边界框面积分布 (像素的平方根)')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'bbox_dimensions_distribution.png')
        plt.close()
        
        # 按类别分析框大小
        plt.figure(figsize=(12, 6))
        df_class_count = df.groupby('class').size().reset_index(name='count')
        
        if len(class_names) > 0:
            df_class_count['class_name'] = df_class_count['class'].apply(
                lambda x: class_names[int(x)] if int(x) < len(class_names) else f'class_{x}'
            )
        else:
            df_class_count['class_name'] = df_class_count['class'].apply(lambda x: f'class_{x}')
        
        # 创建柱状图
        sns.barplot(x='class_name', y='count', data=df_class_count)
        plt.title('每个类别的边界框数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / 'bbox_count_by_class.png')
        plt.close()
        
        # 每个类别的框大小分布 - 只在多个类别时绘制
        if df['class'].nunique() > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 框宽度的箱线图
            sns.boxplot(x='class', y='width', data=df, ax=axes[0])
            axes[0].set_title('各类别的边界框宽度分布')
            axes[0].set_ylim(0, df['width'].quantile(0.95))  # 限制y轴范围以便更好观察
            
            # 框高度的箱线图
            sns.boxplot(x='class', y='height', data=df, ax=axes[1])
            axes[1].set_title('各类别的边界框高度分布')
            axes[1].set_ylim(0, df['height'].quantile(0.95))  # 限制y轴范围以便更好观察
            
            plt.tight_layout()
            plt.savefig(save_dir / 'bbox_dimensions_by_class.png')
            plt.close()
        else:
            # 只有一个类别，保存一个说明文件
            single_class = int(df['class'].iloc[0])
            cls_name = class_names[single_class] if single_class < len(class_names) else f'class_{single_class}'
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"数据集中只有一个类别: {cls_name} (ID: {single_class})", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.savefig(save_dir / 'bbox_dimensions_by_class.png')
            plt.close()
            
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        # 保存一个错误信息文件
        with open(save_dir / 'plotting_error.txt', 'w') as f:
            f.write(f"绘制图表时出错: {str(e)}\n")
            f.write(f"数据统计信息已保存在CSV文件中，可以使用其他工具进行分析。")
    
    return df

def analyze_overlap_boxes(dataset, class_names, save_dir, iou_threshold=0.8):
    """
    分析每张图片中重叠的边界框
    
    参数:
        dataset: YOLODataset实例
        class_names: 类别名称列表
        save_dir: 保存分析结果的目录
        iou_threshold: 判断重叠的IoU阈值
    """
    print("\n== 分析重叠框 ==")
    
    overlap_results = []
    boxes_per_image = []
    boxes_per_class = defaultdict(int)
    
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"处理样本 {idx}/{len(dataset)}")
            
        sample = dataset[idx]
        bboxes = sample['bboxes']  # 归一化的 xywh 格式
        cls = sample['cls']
        img = sample['img']
      
        # 计算每张图片的边界框数量
        boxes_per_image.append(len(bboxes))
        
        # 统计每个类别的框数量
        for c in cls:
            boxes_per_class[int(c)] += 1
        
        # 获取图像尺寸
        img_height, img_width = img.shape[1:3]
        
        # 初始化存储xyxy格式边界框的列表
        xyxy_boxes = []
        
        # 转换边界框格式并绘制
        for box, cls_idx in zip(bboxes, cls):
            # 将xywh转换为xyxy格式
            x, y, w, h = box
            # 注意：YOLODataset返回的边界框坐标已经是相对于原始图像尺寸的归一化坐标
            # 因此需要乘以当前图像尺寸来获得像素坐标
            # 这里的width和height是PIL图像的尺寸，可能与原始图像尺寸不同
            x_center = x * img_width
            y_center = y * img_height
            box_width = w * img_width
            box_height = h * img_height
            
            x1 = int(x_center - box_width/2)
            y1 = int(y_center - box_height/2)
            x2 = int(x_center + box_width/2)
            y2 = int(y_center + box_height/2)
            
            # 获取类别名称和颜色
            cls_id = int(cls_idx)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            color = (255, 0, 0)  # 红色边界框
            
            # 绘制边界框和标签
            # 这里先不绘制，我们只需要收集重叠框信息
            # plot_one_box([x1, y1, x2, y2], img, color=color,
            #            label=f'{cls_name}', line_thickness=2)
            
            # 将XY坐标加入列表，用于后续重叠分析
            xyxy_boxes.append([x1, y1, x2, y2])
            
        # 检查重叠框
        overlap_detected = False
        for i in range(len(xyxy_boxes)):
            for j in range(i+1, len(xyxy_boxes)):
                # 只比较相同类别的框
                if int(cls[i]) == int(cls[j]):
                    iou = calculate_iou(xyxy_boxes[i], xyxy_boxes[j])
                    if iou > iou_threshold:
                        cls_id = int(cls[i])
                        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
                        
                        overlap_results.append({
                            'image_idx': idx,
                            'box1_idx': i,
                            'box2_idx': j,
                            'class': cls_id,
                            'class_name': cls_name,
                            'iou': iou
                        })
                        overlap_detected = True
        
        # 如果发现重叠框，保存图像以便查看
        if overlap_detected and len(overlap_results) <= 20:  # 只保存前20张有重叠的图像
            # 转换图像格式
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # 绘制所有边界框
            for box_idx, box in enumerate(xyxy_boxes):
                cls_id = int(cls[box_idx])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
                
                # 判断是否参与了重叠
                is_overlapped = any(
                    (r['box1_idx'] == box_idx or r['box2_idx'] == box_idx) 
                    and r['image_idx'] == idx 
                    for r in overlap_results
                )
                
                # 重叠框用红色，其他用绿色
                color = (255, 0, 0) if is_overlapped else (0, 255, 0)
                
                plot_one_box(box, img_pil, color=color, 
                            label=f'{cls_name}', line_thickness=2)
            
            # 保存图像
            img_pil.save(save_dir / f'overlap_sample_{idx}.jpg')
    
    # 分析每张图片的框数量
    plt.figure(figsize=(10, 6))
    sns.histplot(boxes_per_image, bins=20, kde=True)
    plt.title('每张图片的边界框数量分布')
    plt.xlabel('边界框数量')
    plt.ylabel('图片数量')
    plt.savefig(save_dir / 'boxes_per_image.png')
    plt.close()
    
    # 检查异常多的框
    percentile_95 = np.percentile(boxes_per_image, 95)
    print(f"每张图片平均框数: {np.mean(boxes_per_image):.2f}")
    print(f"每张图片最大框数: {max(boxes_per_image)}")
    print(f"每张图片框数量95分位数: {percentile_95:.2f}")
    print(f"框数超过95分位数的图片数量: {sum(1 for x in boxes_per_image if x > percentile_95)}")
    
    return boxes_per_image, boxes_per_class

def analyze_class_distribution(dataset, class_names, save_dir):
    """
    分析数据集的类别分布
    
    参数:
        dataset: YOLODataset实例
        class_names: 类别名称列表
        save_dir: 保存分析结果的目录
    """
    print("\n== 分析类别分布 ==")
    
    # 收集所有类别标签
    all_classes = []
    
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"处理样本 {idx}/{len(dataset)}")
            
        sample = dataset[idx]
        cls = sample['cls']
        
        for c in cls:
            all_classes.append(int(c))
    
    # 计算每个类别的数量
    class_counts = Counter(all_classes)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'class_id': list(class_counts.keys()),
        'count': list(class_counts.values())
    }).sort_values('class_id')
    
    # 添加类别名称
    df['class_name'] = df['class_id'].apply(
        lambda x: class_names[int(x)] if int(x) < len(class_names) else f'class_{x}'
    )
    
    # 计算每个类别的百分比
    total = df['count'].sum()
    df['percentage'] = df['count'] / total * 100
    
    # 保存结果
    df.to_csv(save_dir / 'class_distribution.csv', index=False)
    
    # 为避免中文字体问题，创建一个英文的类别标签版本
    df['safe_name'] = df['class_id'].apply(lambda x: f'Class {x}')
    
    # 可视化类别分布
    plt.figure(figsize=(12, 6))
    
    try:
        # 尝试使用中文类别名称
        ax = sns.barplot(x='class_name', y='count', data=df)
        has_chinese_font = True
    except Exception as e:
        print(f"使用中文标签绘图失败: {e}")
        print("使用英文类别标签代替...")
        ax = sns.barplot(x='safe_name', y='count', data=df)
        has_chinese_font = False
    
    # 设置标题和标签
    ax.set_title('数据集类别分布')
    ax.set_xlabel('类别')
    ax.set_ylabel('实例数量')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数量标签
    for i, p in enumerate(ax.patches):
        ax.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')
    
    # 添加图例说明中文类别
    if not has_chinese_font and len(df) <= 10:  # 只有少量类别时添加图例
        legend_elements = []
        for _, row in df.iterrows():
            legend_elements.append(plt.Line2D([0], [0], color='w', marker='s', 
                                           markerfacecolor=sns.color_palette()[int(row['class_id']) % 10], 
                                           markersize=10, label=f"{row['safe_name']}: {row['class_name']}"))
        plt.legend(handles=legend_elements, title="类别对照表", loc='best')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png')
    plt.close()
    
    # 计算类别不平衡度
    max_count = df['count'].max()
    min_count = df['count'].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"总实例数: {total}")
    print(f"类别数: {len(df)}")
    print(f"最多的类别: {df.loc[df['count'].idxmax(), 'class_name']} ({max_count}实例)")
    print(f"最少的类别: {df.loc[df['count'].idxmin(), 'class_name']} ({min_count}实例)")
    print(f"类别不平衡比例 (最多/最少): {imbalance_ratio:.2f}")
    
    # 检查类别是否极度不平衡
    if imbalance_ratio > 10:
        print("警告: 数据集存在极度不平衡!")
    
    return df

def visualize_dataset(data_path, num_samples=16, save_dir='debug_images'):
    """
    可视化数据集中的样本
    
    参数:
        data_path: 数据集配置文件路径
        num_samples: 要可视化的样本数量
        save_dir: 保存可视化结果的目录
    """
    # 设置中文字体支持
    setup_chinese_font()
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集配置
    with open(data_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    # 获取训练集路径
    train_path = data_dict['path'] + data_dict['train_images'].split('.')[1]
    # 创建数据集对象
    dataset = YOLODataset(
        img_path=train_path,
        data=data_dict,
        imgsz=640,
        augment=False,
        cache=False
    )
    
    # 获取类别名称列表
    class_names = data_dict.get('names', [])
    print(f"类别信息: {class_names}")  # 添加调试输出
    
    # ===== 添加新的统计分析 =====
    # 1. 分析边界框尺寸
    stats_dir = save_dir / 'statistics'
    stats_dir.mkdir(exist_ok=True)
    
    bbox_df = analyze_bbox_dimensions(dataset, class_names, stats_dir)
    
    # 2. 分析重叠框
    boxes_per_image, boxes_per_class = analyze_overlap_boxes(dataset, class_names, stats_dir)
    
    # 3. 分析类别分布
    class_df = analyze_class_distribution(dataset, class_names, stats_dir)
    
    # ===== 可视化样本图像 =====
    # 随机选择样本进行可视化
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 创建子图网格
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # 为每个类别分配一个固定的颜色
    num_classes = len(class_names)
    colors = [(np.random.randint(0, 255), 
               np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(max(num_classes, 1))]
    
    for idx, sample_idx in enumerate(indices):
        # 获取样本
        sample = dataset[sample_idx]
        img = sample['img']
        bboxes = sample['bboxes']  # 归一化的 xywh 格式
        cls = sample['cls']
        
        # 转换图像格式
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        # 获取图像尺寸
        width, height = img.size
        
        # 转换边界框格式并绘制
        for box, cls_idx in zip(bboxes, cls):
            # 将xywh转换为xyxy格式
            x, y, w, h = box
            # 注意：YOLODataset返回的边界框坐标已经是相对于原始图像尺寸的归一化坐标
            # 因此需要乘以当前图像尺寸来获得像素坐标
            # 这里的width和height是PIL图像的尺寸，可能与原始图像尺寸不同
            x_center = x * width
            y_center = y * height
            box_width = w * width
            box_height = h * height
            
            x1 = int(x_center - box_width/2)
            y1 = int(y_center - box_height/2)
            x2 = int(x_center + box_width/2)
            y2 = int(y_center + box_height/2)
            
            # 获取类别名称和颜色
            cls_id = int(cls_idx)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            color = colors[cls_id % len(colors)]
            
            # 绘制边界框和标签
            plot_one_box([x1, y1, x2, y2], img, color=color,
                        label=f'{cls_name}', line_thickness=2)
        
        # 将图像转换回numpy数组
        img_np = np.array(img)
        
        # 显示图像
        axes[idx].imshow(img_np)
        axes[idx].axis('off')
        axes[idx].set_title(f'Sample {sample_idx}')
        
        # 同时保存单独的图像文件
        img.save(save_dir / f'sample_{sample_idx}.jpg')
    
    # 移除未使用的子图
    for idx in range(len(indices), len(axes)):
        fig.delaxes(axes[idx])
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_dir / 'dataset_overview.jpg')
    plt.close()
    
    print(f"可视化结果已保存到: {save_dir}")
    print(f"- 总览图: {save_dir/'dataset_overview.jpg'}")
    print(f"- 单张样本: {save_dir}/sample_*.jpg")
    print(f"- 统计分析: {stats_dir}")

if __name__ == '__main__':
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 预先配置中文字体
    print("初始化中文字体支持...")
    setup_chinese_font()
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parents[2]
    
    # 数据集配置文件路径
    data_yaml = project_root / 'database/China_MotorBike/data.yaml'
    
    # 可视化保存路径
    save_dir = project_root / 'improved_yolov8/debug/visualization'
    
    # 运行可视化
    visualize_dataset(str(data_yaml), num_samples=16, save_dir=save_dir) 