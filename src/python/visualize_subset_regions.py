#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本可视化参考图像的前10个POI点的参考子集区域，并使用预设的透视矩阵可视化目标图像中对应的目标子集区域。
修改后使用像素插值方法采集目标子集，保持参考和目标子集大小一致。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 添加src/python目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_python_dir = os.path.join(current_dir, "src", "python")
if os.path.exists(src_python_dir):
    sys.path.append(src_python_dir)
else:
    # 如果当前已经在src/python目录中
    parent_dir = os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "python" and os.path.basename(parent_dir) == "src":
        sys.path.append(current_dir)
    else:
        # 尝试查找实际路径
        print("Warning: Could not find the src/python directory. Please run this script from the project root.")

from python_cubic_interpolation import BicubicBspline, Point2D, Image2D, load_image
from python_nr import POI2D, Subset2D

def main():
    # 获取当前脚本的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置要处理的文件
    ref_image_path = os.path.join(script_dir, "src", "python", "img", "speckle_medium.tif")
    tar_image_path = os.path.join(script_dir, "src", "python", "img", "Camera_DEV_1AB22C0222A5_2025-01-11_10-40-02.png")
    
    # 如果路径不存在，则尝试另一种路径
    if not os.path.exists(ref_image_path):
        ref_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/speckle_medium.tif"
    
    if not os.path.exists(tar_image_path):
        tar_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/Camera_DEV_1AB22C0222A5_2025-01-11_10-40-02.png"     # 替换为您计算机上的路径
    
    # 检查文件是否存在
    if not os.path.exists(ref_image_path) or not os.path.exists(tar_image_path):
        print(f"Error: Image files not found. Please check the paths.")
        print(f"Tried: {ref_image_path}")
        print(f"And: {tar_image_path}")
        return
    
    print(f"Reference image: {ref_image_path}")
    print(f"Target image: {tar_image_path}")
    
    # 加载图像
    try:
        ref_img, ref_cv_img = load_image(ref_image_path)
        tar_img, tar_cv_img = load_image(tar_image_path)
        print(f"Images loaded successfully: {ref_img.width}x{ref_img.height}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # 设置DIC参数
    subset_radius_x = 52
    subset_radius_y = 52
    
    # 设置POI
    upper_left_point = Point2D(975, 695)
    poi_number_x = 9
    poi_number_y = 7
    grid_space = 300
    
    # 创建POI队列
    poi_queue = []
    for i in range(poi_number_y):
        for j in range(poi_number_x):
            x = upper_left_point.x + j * grid_space
            y = upper_left_point.y + i * grid_space
            poi = POI2D(x, y)
            poi_queue.append(poi)
    
    # 预设的透视变换矩阵
    transform_matrix = np.array([
        [ 5.58350133e-01, -1.36953786e-02,  3.63650932e+02], 
        [ 9.52164058e-03,  5.50625148e-01,  5.68499962e+02], 
        [ 3.29165342e-06, -3.78049399e-06,  1.00000000e+00]
    ])
    
    # 创建双三次样条插值器用于目标图像
    tar_interp = BicubicBspline(tar_img)
    tar_interp.prepare()  # 必须调用prepare方法来计算插值系数
    
    # 仅处理前10个POI
    poi_queue = poi_queue[:10]
    
    # 准备可视化
    plt.figure(figsize=(15, 20))
    
    # 设置子图布局
    for i, poi in enumerate(poi_queue):
        # 创建参考子区
        ref_subset = Subset2D(Point2D(poi.x, poi.y), subset_radius_x, subset_radius_y)
        ref_subset.fill(ref_img)
        
        # 计算子集大小
        subset_width = 2 * subset_radius_x + 1
        subset_height = 2 * subset_radius_y + 1
        
        # 使用插值法构建目标子区
        target_subset_data = np.zeros((subset_height, subset_width), dtype=np.float32)
        
        # 计算变换后的目标点位置（中心点）
        pt = np.array([poi.x, poi.y, 1.0])
        transformed = np.dot(transform_matrix, pt)
        transformed /= transformed[2]  # 归一化
        target_x, target_y = transformed[0], transformed[1]
        
        # 遍历子集中的每个像素，应用变换并进行插值
        for r in range(subset_height):
            for c in range(subset_width):
                # 计算相对于中心点的局部坐标
                x_local = c - subset_radius_x
                y_local = r - subset_radius_y
                
                # 构建参考图像中的全局点
                ref_global_x = poi.x + x_local
                ref_global_y = poi.y + y_local
                
                # 应用透视变换计算这个点在目标图像中的位置
                ref_pt = np.array([ref_global_x, ref_global_y, 1.0])
                tar_pt = np.dot(transform_matrix, ref_pt)
                tar_pt /= tar_pt[2]  # 归一化
                
                # 通过插值获取目标图像中的像素值
                target_subset_data[r, c] = tar_interp.compute(Point2D(tar_pt[0], tar_pt[1]))
        
        # 绘制参考子区
        plt.subplot(10, 3, i*3 + 1)
        plt.imshow(ref_subset.eg_mat, cmap='gray')
        plt.title(f"Ref #{i+1}: ({poi.x:.1f}, {poi.y:.1f})")
        plt.axis('off')
        
        # 绘制目标子区
        plt.subplot(10, 3, i*3 + 2)
        plt.imshow(target_subset_data, cmap='gray')
        plt.title(f"Target #{i+1}: ({target_x:.1f}, {target_y:.1f})")
        plt.axis('off')
        
        # 绘制差异图
        plt.subplot(10, 3, i*3 + 3)
        # 计算差异
        ref_norm = cv2.normalize(ref_subset.eg_mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tar_norm = cv2.normalize(target_subset_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff = np.abs(ref_norm - tar_norm)
        
        # 创建伪彩色差异图
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        plt.imshow(cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Difference #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_path = "subset_visualization_interpolated.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # 显示图像
    plt.show()
    
if __name__ == "__main__":
    main()