#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个示例演示如何使用Python版本的OpenCorr库实现一个NR算法
（使用一阶形函数）的路径无关的DIC方法。
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from python_cubic_interpolation import BicubicBspline, Point2D, Image2D, load_image
from python_nr import NR2D1, POI2D, Deformation2D1, Subset2D
from python_gradient import Gradient2D4

# 为NR2D1类添加一个可视化子区的方法
class VisualizedNR2D1(NR2D1):
    """增强的NR2D1类，添加了可视化功能"""
    def __init__(self, subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size=16, 
                 visualize=False, visualize_interval=5, poi_index=0, total_pois=0):
        super().__init__(subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size)
        self.visualize = visualize
        self.visualize_interval = visualize_interval
        self.visualization_window = None
        self.poi_index = poi_index
        self.total_pois = total_pois
    
    def compute(self, poi, poi_index=None, total_pois=None):
        """重写compute方法，增加可视化功能"""
        if poi_index is not None:
            self.poi_index = poi_index
        if total_pois is not None:
            self.total_pois = total_pois
            
        subset_width = 2 * self.subset_radius_x + 1
        subset_height = 2 * self.subset_radius_y + 1
        
        # 检查POI是否有效
        if (poi.y - self.subset_radius_y < 0 or poi.x - self.subset_radius_x < 0 or
            poi.y + self.subset_radius_y > self.ref_img.height - 1 or 
            poi.x + self.subset_radius_x > self.ref_img.width - 1 or
            abs(poi.deformation.u) >= self.ref_img.width or 
            abs(poi.deformation.v) >= self.ref_img.height or
            poi.result.zncc < 0 or np.isnan(poi.deformation.u) or 
            np.isnan(poi.deformation.v)):
            
            poi.result.zncc = poi.result.zncc if poi.result.zncc < -1 else -1
            return
        
        # 创建参考子区
        ref_subset = Subset2D(Point2D(poi.x, poi.y), self.subset_radius_x, self.subset_radius_y)
        ref_subset.fill(self.ref_img)
        ref_mean = np.mean(ref_subset.eg_mat)
        ref_subset_visual = ref_subset.eg_mat.copy()  # 保存可视化用的原始子区
        ref_mean_norm = ref_subset.zeroMeanNorm()
        
        # 创建目标子区
        tar_subset = Subset2D(Point2D(poi.x, poi.y), self.subset_radius_x, self.subset_radius_y)
        
        # 获取初始变形参数
        p_initial = Deformation2D1(
            poi.deformation.u, poi.deformation.ux, poi.deformation.uy,
            poi.deformation.v, poi.deformation.vx, poi.deformation.vy
        )
        
        # 初始化迭代
        iteration_counter = 0
        p_current = Deformation2D1()
        p_current.setDeformation(p_initial)
        p_numba = p_current.to_numba()  # 转换为numba类型用于加速计算
        
        # 为计算准备数据结构
        tar_gradient_x = np.zeros((subset_height, subset_width), dtype=np.float32)
        tar_gradient_y = np.zeros((subset_height, subset_width), dtype=np.float32)
        
        # Newton-Raphson迭代
        dp_norm_max = 0.0
        znssd = 0.0

        # 准备可视化窗口
        if self.visualize:
            window_title = f"Subset Matching - POI #{self.poi_index+1}/{self.total_pois} at ({poi.x:.1f}, {poi.y:.1f})"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            # 放大窗口尺寸到原来的2倍
            display_scale = 5.0  # 缩放因子
            display_width = int(subset_width * 4 * display_scale)
            display_height = int(subset_height * 2 * display_scale)
            cv2.resizeWindow(window_title, display_width, display_height)
        
        while True:
            iteration_counter += 1
            
            # 重构目标子区
            for r in range(subset_height):
                for c in range(subset_width):
                    x_local = c - self.subset_radius_x
                    y_local = r - self.subset_radius_y
                    
                    # 计算变形后的坐标
                    point = Point2D(x_local, y_local)
                    warped = p_current.warp(point)
                    global_x = poi.x + warped.x
                    global_y = poi.y + warped.y
                    
                    # 计算插值
                    global_point = Point2D(global_x, global_y)
                    tar_subset.eg_mat[r, c] = self.tar_interp.compute(global_point)
                    tar_gradient_x[r, c] = self.tar_interp_x.compute(global_point)
                    tar_gradient_y[r, c] = self.tar_interp_y.compute(global_point)
            
            # 保存可视化用的原始目标子区
            tar_subset_visual = tar_subset.eg_mat.copy()
            
            # 计算目标子区的零均值归一化
            tar_mean = np.mean(tar_subset.eg_mat)
            tar_subset.eg_mat = tar_subset.eg_mat - tar_mean
            tar_mean_norm = np.sqrt(np.sum(tar_subset.eg_mat * tar_subset.eg_mat))
            
            # 可视化当前迭代的子区匹配
            if self.visualize:
                # 放大子区图像
                scale_factor = 5.0  # 图像放大因子
                
                # 归一化子区图像用于显示
                ref_norm = cv2.normalize(ref_subset_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                tar_norm = cv2.normalize(tar_subset_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # 计算差异图
                diff = np.abs(ref_subset_visual - tar_subset_visual)
                diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # 放大图像
                ref_norm = cv2.resize(ref_norm, (int(ref_norm.shape[1] * scale_factor), int(ref_norm.shape[0] * scale_factor)))
                tar_norm = cv2.resize(tar_norm, (int(tar_norm.shape[1] * scale_factor), int(tar_norm.shape[0] * scale_factor)))
                diff_norm = cv2.resize(diff_norm, (int(diff_norm.shape[1] * scale_factor), int(diff_norm.shape[0] * scale_factor)))
                
                # 创建伪彩色差异图
                diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
                
                # 将灰度图转为彩色以便显示
                ref_color = cv2.cvtColor(ref_norm, cv2.COLOR_GRAY2BGR)
                tar_color = cv2.cvtColor(tar_norm, cv2.COLOR_GRAY2BGR)
                
                # 创建显示图像
                top_row = np.hstack((ref_color, tar_color))
                bottom_row = np.hstack((diff_color, np.zeros_like(diff_color)))  # 右下角空白
                
                display = np.vstack((top_row, bottom_row))
                
                # 调整字体大小
                font_size = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                title_color = (0, 255, 255)
                text_offset_x = 20  # 横向偏移
                
                # 计算放大后的子区宽度以便正确放置文字
                scaled_subset_width = int(subset_width * scale_factor)
                scaled_subset_height = int(subset_height * scale_factor)
                
                # 添加文字信息
                cv2.putText(display, f"Reference Subset", (text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                cv2.putText(display, f"Target Subset", (scaled_subset_width + text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                cv2.putText(display, f"Difference Map", (text_offset_x, scaled_subset_height + 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                
                # 数据信息放在右下角
                info_x = scaled_subset_width + text_offset_x
                info_y_start = scaled_subset_height + 60
                line_spacing = 35
                
                cv2.putText(display, f"POI #{self.poi_index+1}/{self.total_pois}", 
                            (info_x, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"Position: ({poi.x:.1f}, {poi.y:.1f})", 
                            (info_x, info_y_start + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"Iteration: {iteration_counter}", 
                            (info_x, info_y_start + line_spacing*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"ZNCC: {0.5 * (2 - znssd):.4f}", 
                            (info_x, info_y_start + line_spacing*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"u: {p_current.u:.2f}, v: {p_current.v:.2f}", 
                            (info_x, info_y_start + line_spacing*4), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                
                # 命令提示
                cv2.putText(display, "Press 'c' to skip to next POI, ESC to exit", 
                           (text_offset_x, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), font_thickness)
                
                # 显示图像
                cv2.imshow(window_title, display)
                key = cv2.waitKey(5)  # 增加等待时间，从1ms到5ms
                if key == 27:  # ESC键退出整个程序
                    cv2.destroyAllWindows()
                    import sys
                    sys.exit(0)
                elif key == ord('c'):  # 'c'键跳过当前POI继续下一个
                    break
            
            # 使用numba加速计算黑塞矩阵和最速下降图像
            hessian = np.zeros((6, 6), dtype=np.float32)
            sd_img = np.zeros((subset_height, subset_width, 6), dtype=np.float32)
            
            for r in range(subset_height):
                for c in range(subset_width):
                    x_local = c - self.subset_radius_x
                    y_local = r - self.subset_radius_y
                    tar_grad_x = tar_gradient_x[r, c]
                    tar_grad_y = tar_gradient_y[r, c]
                    
                    # 构建最速下降图像
                    sd_img[r, c, 0] = tar_grad_x
                    sd_img[r, c, 1] = tar_grad_x * x_local
                    sd_img[r, c, 2] = tar_grad_x * y_local
                    sd_img[r, c, 3] = tar_grad_y
                    sd_img[r, c, 4] = tar_grad_y * x_local
                    sd_img[r, c, 5] = tar_grad_y * y_local
                    
                    # 计算黑塞矩阵元素
                    for i in range(6):
                        for j in range(6):
                            hessian[i, j] += sd_img[r, c, i] * sd_img[r, c, j]
            
            # 计算黑塞矩阵的逆
            inv_hessian = np.linalg.inv(hessian)
            
            # 计算误差图像
            error_img = ref_subset.eg_mat * (tar_mean_norm / ref_mean_norm) - tar_subset.eg_mat
            
            # 计算ZNSSD
            znssd = np.sum(error_img * error_img) / (tar_mean_norm * tar_mean_norm)
            
            # 计算参数增量的分子部分
            numerator = np.zeros(6, dtype=np.float32)
            for r in range(subset_height):
                for c in range(subset_width):
                    for i in range(6):
                        numerator[i] += sd_img[r, c, i] * error_img[r, c]
            
            # 计算参数增量
            dp = np.zeros(6, dtype=np.float32)
            for i in range(6):
                for j in range(6):
                    dp[i] += inv_hessian[i, j] * numerator[j]
            
            # 更新当前参数
            p_current.setDeformation(
                p_current.u + dp[0],
                p_current.ux + dp[1],
                p_current.uy + dp[2],
                p_current.v + dp[3],
                p_current.vx + dp[4],
                p_current.vy + dp[5]
            )
            
            # 检查收敛
            subset_radius_x2 = self.subset_radius_x * self.subset_radius_x
            subset_radius_y2 = self.subset_radius_y * self.subset_radius_y
            
            dp_norm_max = (dp[0] * dp[0] + 
                        dp[1] * dp[1] * subset_radius_x2 +
                        dp[2] * dp[2] * subset_radius_y2 +
                        dp[3] * dp[3] +
                        dp[4] * dp[4] * subset_radius_x2 +
                        dp[5] * dp[5] * subset_radius_y2)
            
            dp_norm_max = np.sqrt(dp_norm_max)
            
            # 检查是否满足终止条件
            if iteration_counter >= self.stop_condition or dp_norm_max < self.conv_criterion:
                # 如果已收敛，再次显示最终结果
                if self.visualize:
                    # 添加收敛信息
                    cv2.rectangle(display, (0, display.shape[0]-60), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                    cv2.putText(display, "CONVERGED - Press any key to continue", 
                                (display.shape[1]//2 - 250, display.shape[0]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(window_title, display)
                    cv2.waitKey(0)  # 等待用户按键继续
                break
        
        # 关闭可视化窗口
        if self.visualize:
            cv2.destroyWindow(window_title)
        
        # 存储最终结果
        poi.deformation.u = p_current.u
        poi.deformation.ux = p_current.ux
        poi.deformation.uy = p_current.uy
        poi.deformation.v = p_current.v
        poi.deformation.vx = p_current.vx
        poi.deformation.vy = p_current.vy
        
        # 保存输出参数
        poi.result.u0 = p_initial.u
        poi.result.v0 = p_initial.v
        poi.result.zncc = 0.5 * (2 - znssd)
        poi.result.iteration = float(iteration_counter)
        poi.result.convergence = dp_norm_max
        
        # 检查迭代是否在期望目标处收敛
        if poi.result.convergence >= self.conv_criterion and poi.result.iteration >= self.stop_condition:
            poi.result.zncc = -4.0
        
        # 检查是否出现NaN
        if (np.isnan(poi.result.zncc) or np.isnan(poi.deformation.u) or np.isnan(poi.deformation.v)):
            poi.deformation.u = poi.result.u0
            poi.deformation.v = poi.result.v0
            poi.result.zncc = -5.0
        
        return poi

class VisualizedNR2D1WithPerspective(VisualizedNR2D1):
    """增强的NR2D1类，使用透视变换构建目标子集，同时保留可视化功能"""
    def __init__(self, subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size=16, 
                 visualize=False, visualize_interval=5, poi_index=0, total_pois=0, transform_matrix=None):
        super().__init__(subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size,
                         visualize, visualize_interval, poi_index, total_pois)
        self.transform_matrix = transform_matrix
        
    def set_transform_matrix(self, matrix):
        """设置透视变换矩阵"""
        self.transform_matrix = matrix
        
    def compute(self, poi, poi_index=None, total_pois=None):
        """使用透视变换构建目标子集，而不是变形参数"""
        if poi_index is not None:
            self.poi_index = poi_index
        if total_pois is not None:
            self.total_pois = total_pois
            
        subset_width = 2 * self.subset_radius_x + 1
        subset_height = 2 * self.subset_radius_y + 1
        
        # 检查POI是否有效 (保持原有代码)
        if (poi.y - self.subset_radius_y < 0 or poi.x - self.subset_radius_x < 0 or
            poi.y + self.subset_radius_y > self.ref_img.height - 1 or 
            poi.x + self.subset_radius_x > self.ref_img.width - 1 or
            abs(poi.deformation.u) >= self.ref_img.width or 
            abs(poi.deformation.v) >= self.ref_img.height or
            poi.result.zncc < 0 or np.isnan(poi.deformation.u) or 
            np.isnan(poi.deformation.v)):
            
            poi.result.zncc = poi.result.zncc if poi.result.zncc < -1 else -1
            return
        
        # 创建参考子区 (保持原有代码)
        ref_subset = Subset2D(Point2D(poi.x, poi.y), self.subset_radius_x, self.subset_radius_y)
        ref_subset.fill(self.ref_img)
        ref_mean = np.mean(ref_subset.eg_mat)
        ref_subset_visual = ref_subset.eg_mat.copy()
        ref_mean_norm = ref_subset.zeroMeanNorm()
        
        # 创建目标子区
        tar_subset = Subset2D(Point2D(poi.x, poi.y), self.subset_radius_x, self.subset_radius_y)
        
        # 获取初始变形参数 (保持原有代码)
        p_initial = Deformation2D1(
            poi.deformation.u, poi.deformation.ux, poi.deformation.uy,
            poi.deformation.v, poi.deformation.vx, poi.deformation.vy
        )
        
        # 初始化迭代 (保持原有代码)
        iteration_counter = 0
        p_current = Deformation2D1()
        p_current.setDeformation(p_initial)
        
        # 为计算准备数据结构
        tar_gradient_x = np.zeros((subset_height, subset_width), dtype=np.float32)
        tar_gradient_y = np.zeros((subset_height, subset_width), dtype=np.float32)
        
        dp_norm_max = 0.0
        znssd = 0.0
        
        # 准备可视化窗口 (保持原有代码)
        if self.visualize:
            window_title = f"Subset Matching - POI #{self.poi_index+1}/{self.total_pois} at ({poi.x:.1f}, {poi.y:.1f})"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            display_scale = 5.0
            display_width = int(subset_width * 4 * display_scale)
            display_height = int(subset_height * 2 * display_scale)
            cv2.resizeWindow(window_title, display_width, display_height)
        
        while True:
            iteration_counter += 1
            
            # ------ 替换开始: 使用透视变换构建目标子区 ------
            
            # 从当前变形参数构建临时透视变换矩阵
            if self.transform_matrix is not None and iteration_counter == 1:
                # 第一次迭代时使用全局透视变换矩阵作为初始估计
                transform = self.transform_matrix.copy()
            else:
                # 使用当前变形参数创建局部仿射变换矩阵
                transform = np.array([
                    [1.0 + p_current.ux, p_current.uy, p_current.u],
                    [p_current.vx, 1.0 + p_current.vy, p_current.v],
                    [0, 0, 1.0]
                ], dtype=np.float32)
            
            # 遍历子集中的每个像素，应用变换并进行插值
            for r in range(subset_height):
                for c in range(subset_width):
                    # 计算相对于中心点的局部坐标
                    x_local = c - self.subset_radius_x
                    y_local = r - self.subset_radius_y
                    
                    # 构建参考图像中的全局点
                    ref_global_x = poi.x + x_local
                    ref_global_y = poi.y + y_local
                    
                    # 应用透视变换计算这个点在目标图像中的位置
                    ref_pt = np.array([ref_global_x, ref_global_y, 1.0])
                    tar_pt = np.dot(transform, ref_pt)
                    tar_pt /= tar_pt[2]  # 归一化
                    
                    # 通过插值获取目标图像中的像素值和梯度
                    global_point = Point2D(tar_pt[0], tar_pt[1])
                    tar_subset.eg_mat[r, c] = self.tar_interp.compute(global_point)
                    tar_gradient_x[r, c] = self.tar_interp_x.compute(global_point)
                    tar_gradient_y[r, c] = self.tar_interp_y.compute(global_point)
            
            # ------ 替换结束 ------
            
            # 保存可视化用的原始目标子区
            tar_subset_visual = tar_subset.eg_mat.copy()
            
            # 计算目标子区的零均值归一化
            tar_mean = np.mean(tar_subset.eg_mat)
            tar_subset.eg_mat = tar_subset.eg_mat - tar_mean
            tar_mean_norm = np.sqrt(np.sum(tar_subset.eg_mat * tar_subset.eg_mat))
            
            # 可视化当前迭代的子区匹配 (保持原有代码)
            if self.visualize:
                # 放大子区图像
                scale_factor = 5.0  # 图像放大因子
                
                # 归一化子区图像用于显示
                ref_norm = cv2.normalize(ref_subset_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                tar_norm = cv2.normalize(tar_subset_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # 计算差异图
                diff = np.abs(ref_subset_visual - tar_subset_visual)
                diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # 放大图像
                ref_norm = cv2.resize(ref_norm, (int(ref_norm.shape[1] * scale_factor), int(ref_norm.shape[0] * scale_factor)))
                tar_norm = cv2.resize(tar_norm, (int(tar_norm.shape[1] * scale_factor), int(tar_norm.shape[0] * scale_factor)))
                diff_norm = cv2.resize(diff_norm, (int(diff_norm.shape[1] * scale_factor), int(diff_norm.shape[0] * scale_factor)))
                
                # 创建伪彩色差异图
                diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
                
                # 将灰度图转为彩色以便显示
                ref_color = cv2.cvtColor(ref_norm, cv2.COLOR_GRAY2BGR)
                tar_color = cv2.cvtColor(tar_norm, cv2.COLOR_GRAY2BGR)
                
                # 创建显示图像
                top_row = np.hstack((ref_color, tar_color))
                bottom_row = np.hstack((diff_color, np.zeros_like(diff_color)))  # 右下角空白
                
                display = np.vstack((top_row, bottom_row))
                
                # 调整字体大小
                font_size = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                title_color = (0, 255, 255)
                text_offset_x = 20  # 横向偏移
                
                # 计算放大后的子区宽度以便正确放置文字
                scaled_subset_width = int(subset_width * scale_factor)
                scaled_subset_height = int(subset_height * scale_factor)
                
                # 添加文字信息
                cv2.putText(display, f"Reference Subset", (text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                cv2.putText(display, f"Target Subset", (scaled_subset_width + text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                cv2.putText(display, f"Difference Map", (text_offset_x, scaled_subset_height + 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, title_color, font_thickness)
                
                # 数据信息放在右下角
                info_x = scaled_subset_width + text_offset_x
                info_y_start = scaled_subset_height + 60
                line_spacing = 35
                
                cv2.putText(display, f"POI #{self.poi_index+1}/{self.total_pois}", 
                            (info_x, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"Position: ({poi.x:.1f}, {poi.y:.1f})", 
                            (info_x, info_y_start + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"Iteration: {iteration_counter}", 
                            (info_x, info_y_start + line_spacing*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"ZNCC: {0.5 * (2 - znssd):.4f}", 
                            (info_x, info_y_start + line_spacing*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                cv2.putText(display, f"u: {p_current.u:.2f}, v: {p_current.v:.2f}", 
                            (info_x, info_y_start + line_spacing*4), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
                
                # 命令提示
                cv2.putText(display, "Press 'c' to skip to next POI, ESC to exit", 
                           (text_offset_x, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), font_thickness)
                
                # 显示图像
                cv2.imshow(window_title, display)
                key = cv2.waitKey(5)  # 增加等待时间，从1ms到5ms
                if key == 27:  # ESC键退出整个程序
                    cv2.destroyAllWindows()
                    import sys
                    sys.exit(0)
                elif key == ord('c'):  # 'c'键跳过当前POI继续下一个
                    break
            
            # 使用numba加速计算黑塞矩阵和最速下降图像
            hessian = np.zeros((6, 6), dtype=np.float32)
            sd_img = np.zeros((subset_height, subset_width, 6), dtype=np.float32)
            
            for r in range(subset_height):
                for c in range(subset_width):
                    x_local = c - self.subset_radius_x
                    y_local = r - self.subset_radius_y
                    tar_grad_x = tar_gradient_x[r, c]
                    tar_grad_y = tar_gradient_y[r, c]
                    
                    # 构建最速下降图像
                    sd_img[r, c, 0] = tar_grad_x
                    sd_img[r, c, 1] = tar_grad_x * x_local
                    sd_img[r, c, 2] = tar_grad_x * y_local
                    sd_img[r, c, 3] = tar_grad_y
                    sd_img[r, c, 4] = tar_grad_y * x_local
                    sd_img[r, c, 5] = tar_grad_y * y_local
                    
                    # 计算黑塞矩阵元素
                    for i in range(6):
                        for j in range(6):
                            hessian[i, j] += sd_img[r, c, i] * sd_img[r, c, j]
            
            # 计算黑塞矩阵的逆
            inv_hessian = np.linalg.inv(hessian)
            
            # 计算误差图像
            error_img = ref_subset.eg_mat * (tar_mean_norm / ref_mean_norm) - tar_subset.eg_mat
            
            # 计算ZNSSD
            znssd = np.sum(error_img * error_img) / (tar_mean_norm * tar_mean_norm)
            
            # 计算参数增量的分子部分
            numerator = np.zeros(6, dtype=np.float32)
            for r in range(subset_height):
                for c in range(subset_width):
                    for i in range(6):
                        numerator[i] += sd_img[r, c, i] * error_img[r, c]
            
            # 计算参数增量
            dp = np.zeros(6, dtype=np.float32)
            for i in range(6):
                for j in range(6):
                    dp[i] += inv_hessian[i, j] * numerator[j]
            
            # 更新当前参数
            p_current.setDeformation(
                p_current.u + dp[0],
                p_current.ux + dp[1],
                p_current.uy + dp[2],
                p_current.v + dp[3],
                p_current.vx + dp[4],
                p_current.vy + dp[5]
            )
            
            # 检查收敛
            subset_radius_x2 = self.subset_radius_x * self.subset_radius_x
            subset_radius_y2 = self.subset_radius_y * self.subset_radius_y
            
            dp_norm_max = (dp[0] * dp[0] + 
                        dp[1] * dp[1] * subset_radius_x2 +
                        dp[2] * dp[2] * subset_radius_y2 +
                        dp[3] * dp[3] +
                        dp[4] * dp[4] * subset_radius_x2 +
                        dp[5] * dp[5] * subset_radius_y2)
            
            dp_norm_max = np.sqrt(dp_norm_max)
            
            # 检查是否满足终止条件
            if iteration_counter >= self.stop_condition or dp_norm_max < self.conv_criterion:
                # 如果已收敛，再次显示最终结果
                if self.visualize:
                    # 添加收敛信息
                    cv2.rectangle(display, (0, display.shape[0]-60), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                    cv2.putText(display, "CONVERGED - Press any key to continue", 
                                (display.shape[1]//2 - 250, display.shape[0]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(window_title, display)
                    cv2.waitKey(0)  # 等待用户按键继续
                break
        
        # 关闭可视化窗口
        if self.visualize:
            cv2.destroyWindow(window_title)
        
        # 存储最终结果
        poi.deformation.u = p_current.u
        poi.deformation.ux = p_current.ux
        poi.deformation.uy = p_current.uy
        poi.deformation.v = p_current.v
        poi.deformation.vx = p_current.vx
        poi.deformation.vy = p_current.vy
        
        # 保存输出参数
        poi.result.u0 = p_initial.u
        poi.result.v0 = p_initial.v
        poi.result.zncc = 0.5 * (2 - znssd)
        poi.result.iteration = float(iteration_counter)
        poi.result.convergence = dp_norm_max
        
        # 检查迭代是否在期望目标处收敛
        if poi.result.convergence >= self.conv_criterion and poi.result.iteration >= self.stop_condition:
            poi.result.zncc = -4.0
        
        # 检查是否出现NaN
        if (np.isnan(poi.result.zncc) or np.isnan(poi.deformation.u) or np.isnan(poi.deformation.v)):
            poi.deformation.u = poi.result.u0
            poi.deformation.v = poi.result.v0
            poi.result.zncc = -5.0
        
        return poi

# 使用透视变换矩阵W计算初始变形预测值
class PerspectiveInitializer:
    """使用透视变换计算初始变形预测值"""
    def __init__(self, thread_number=4):
        self.thread_number = thread_number
        self.ref_img = None
        self.tar_img = None
        self.transform_matrix = None
    
    def setImages(self, ref_img, tar_img):
        """设置参考图像和目标图像"""
        self.ref_img = ref_img
        self.tar_img = tar_img
    
    def setTransformMatrix(self, matrix=None):
        """设置或自动估计变换矩阵"""
        if matrix is not None:
            # 使用提供的矩阵
            self.transform_matrix = matrix
            print("Using provided perspective transform matrix.")
            return
        
        # 如果没有提供矩阵，尝试使用特征点匹配来估计透视变换
        try:
            print("Attempting to estimate perspective transform from images...")
            # 使用SIFT检测特征点
            sift = cv2.SIFT_create()
            
            # 检测关键点和计算描述符
            ref_gray = self.ref_img.eg_mat.astype(np.uint8)
            tar_gray = self.tar_img.eg_mat.astype(np.uint8)
            
            kp1, des1 = sift.detectAndCompute(ref_gray, None)
            kp2, des2 = sift.detectAndCompute(tar_gray, None)
            
            # FLANN特征匹配
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # 应用比率测试，筛选好的匹配点
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 4:
                print(f"Warning: Not enough good matches found ({len(good_matches)}). Using identity matrix.")
                self.transform_matrix = np.eye(3)
                return
            
            # 获取匹配点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算透视变换矩阵
            self.transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            print(f"Estimated perspective transform from {len(good_matches)} matched features.")
            
            # 可以显示匹配结果用于调试
            # matchesMask = mask.ravel().tolist()
            # draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            # img3 = cv2.drawMatches(ref_gray, kp1, tar_gray, kp2, good_matches, None, **draw_params)
            # plt.figure(figsize=(10, 6))
            # plt.imshow(img3)
            # plt.title('Matched Features')
            # plt.show()
            
        except Exception as e:
            print(f"Error estimating transform: {e}")
            print("Using identity matrix instead.")
            self.transform_matrix = np.eye(3)
    
    def compute(self, poi_queue):
        """为POI计算初始变形预测值"""
        if self.transform_matrix is None:
            self.setTransformMatrix()
        
        print(f"Computing initial deformation prediction using perspective transform for {len(poi_queue)} POIs...")
        
        # 透视变换矩阵
        H = self.transform_matrix
        
        # 首先对H进行归一化，使得H[2,2] = 1
        W = H / H[2, 2]
        
        for poi in poi_queue:
            # 原始坐标
            x, y = poi.x, poi.y
            
            # 应用透视变换得到新坐标
            pt = np.array([x, y, 1.0])
            transformed = np.dot(H, pt)
            transformed /= transformed[2]  # 归一化
            new_x, new_y = transformed[0], transformed[1]
            
            # 直接从透视函数矩阵W中提取变形参数
            # 根据给定格式: W(x,y;p) = s[x' y' 1]^T = [1+ux uy u; vx 1+vy v; uxy vxy 1][x y 1]^T
            
            # 从W矩阵中提取变形参数
            ux = W[0, 0] - 1.0       # 对应 1+ux 中的 ux
            uy = W[0, 1]             # 对应 uy
            u = W[0, 2]              # 对应 u
            vx = W[1, 0]             # 对应 vx
            vy = W[1, 1] - 1.0       # 对应 1+vy 中的 vy
            v = W[1, 2]              # 对应 v
            
            # 存储结果
            poi.deformation.u = float(u)
            poi.deformation.v = float(v)
            poi.deformation.ux = float(ux)
            poi.deformation.uy = float(uy)
            poi.deformation.vx = float(vx)
            poi.deformation.vy = float(vy)
            poi.result.zncc = 1.0    # 设为正值以表示有效
            poi.result.u0 = float(u) # 记录初始猜测值
            poi.result.v0 = float(v) # 记录初始猜测值
        
        print("Perspective transform prediction completed.")
        return poi_queue

def main():
    # 设置要处理的文件
    ref_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/speckle_medium.tif"  # 替换为您计算机上的路径
    tar_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/Camera_DEV_1AB22C0222A5_2025-01-11_10-40-02.png"     # 替换为您计算机上的路径
    
    # 检查文件是否存在
    if not os.path.exists(ref_image_path) or not os.path.exists(tar_image_path):
        print(f"Error: Image files not found. Please check the paths.")
        return
    
    # 加载图像
    try:
        ref_img, ref_cv_img = load_image(ref_image_path)
        tar_img, tar_cv_img = load_image(tar_image_path)
        print(f"Images loaded successfully: {ref_img.width}x{ref_img.height}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # 初始化计时参数
    computation_time = []
    
    # 初始化时间
    timer_tic = time.time()
    
    # 设置输出文件路径
    base_path = os.path.splitext(tar_image_path)[0]
    results_csv_path = f"{base_path}_perspective_nr1_r16.csv"
    time_csv_path = f"{base_path}_perspective_nr1_r16_time.csv"
    img_result_path = f"{base_path}_perspective_nr1_r16_results.png"
    
    # 设置DIC参数
    subset_radius_x = 50
    subset_radius_y = 50
    max_iteration = 100
    max_deformation_norm = 0.001
    
    # 设置POI
    upper_left_point = Point2D(975, 695)
    poi_number_x = 9  # 减少点数以加快示例运行速度
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
    
    # 创建算法实例
    # 获取可用CPU线程数
    import multiprocessing
    cpu_thread_number = max(1, multiprocessing.cpu_count() - 1)
    
    # 创建透视变换初始化器实例
    persp_init = PerspectiveInitializer(cpu_thread_number)
    persp_init.setImages(ref_img, tar_img)
    
    # 可选：手动设置透视变换矩阵
    # 这是一个示例矩阵，表示轻微的平移和缩放，实际应用中应替换为实际变换
    transform_matrix = np.array([
        [ 5.58350133e-01, -1.36953786e-02,  3.63650932e+02], 
        [ 9.52164058e-03,  5.50625148e-01,  5.68499962e+02], 
        [ 3.29165342e-06, -3.78049399e-06,  1.00000000e+00]
    ])
    persp_init.setTransformMatrix(transform_matrix)
    
    # 创建可视化NR实例
    nr = VisualizedNR2D1WithPerspective(subset_radius_x, subset_radius_y, max_deformation_norm, max_iteration, 
                         batch_size=16, visualize=True, visualize_interval=5, transform_matrix=transform_matrix)
    nr.set_images(ref_img, tar_img)
    
    # 记录初始化时间
    timer_toc = time.time()
    init_time = timer_toc - timer_tic
    computation_time.append(init_time)
    
    print(f"Initialization with {len(poi_queue)} POIs takes {init_time:.3f} sec, {cpu_thread_number} CPU threads launched.")
    
    # 透视变换预测初始变形
    timer_tic = time.time()
    persp_init.compute(poi_queue)
    timer_toc = time.time()
    persp_time = timer_toc - timer_tic
    computation_time.append(persp_time)
    
    print(f"Initial deformation prediction using perspective transform takes {persp_time:.3f} sec.")
    
    # NR 计算
    timer_tic = time.time()
    # 使用内置的prepare方法而不是手动计算
    nr.prepare()
    
    # 为了对每个POI点都可视化，我们改用单个POI的方式处理，而不是批处理
    print("Processing all POIs with visualization...")
    print("Press 'c' to skip to next POI, any other key to continue, ESC to exit program.")
    nr.visualize = True
    nr.total_pois = len(poi_queue)
    
    for i, poi in enumerate(poi_queue):
        print(f"Processing POI {i+1}/{len(poi_queue)} at position ({poi.x}, {poi.y})...")
        nr.poi_index = i
        nr.compute(poi)
        
        # 每5个POI显示一次当前的处理进度百分比
        if (i+1) % 5 == 0 or i+1 == len(poi_queue):
            progress = ((i+1) / len(poi_queue)) * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{len(poi_queue)} POIs processed)")
    
    print("All POIs processed successfully with visualization.")
    
    timer_toc = time.time()
    nr_time = timer_toc - timer_tic
    computation_time.append(nr_time)
    
    print(f"Deformation determination using NR takes {nr_time:.3f} sec.")
    
    # 保存计算结果
    try:
        # 准备结果数据
        results = []
        for poi in poi_queue:
            results.append({
                'x': poi.x,
                'y': poi.y,
                'u': poi.deformation.u,
                'v': poi.deformation.v,
                'ux': poi.deformation.ux,
                'uy': poi.deformation.uy,
                'vx': poi.deformation.vx,
                'vy': poi.deformation.vy,  # 添加变形梯度参数
                'zncc': poi.result.zncc,
                'iterations': poi.result.iteration,
                'convergence': poi.result.convergence
            })
        
        # 创建DataFrame并保存
        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv_path, index=False)
        
        # 保存计算时间
        df_time = pd.DataFrame({
            'POI number': [len(poi_queue)],
            'Initialization': [computation_time[0]],
            'Perspective Transform': [computation_time[1]],
            'NR': [computation_time[2]]
        })
        df_time.to_csv(time_csv_path, index=False)
        
        print(f"Results saved to {results_csv_path}")
        print(f"Computation time saved to {time_csv_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # 可视化一些结果
    try:
        plt.figure(figsize=(12, 10))
        
        # 绘制参考图像
        plt.subplot(2, 3, 1)
        plt.imshow(ref_cv_img, cmap='gray')
        plt.title("Reference Image")
        plt.axis('off')
        
        # 绘制目标图像
        plt.subplot(2, 3, 2)
        plt.imshow(tar_cv_img, cmap='gray')
        plt.title("Target Image")
        plt.axis('off')
        
        # 绘制位移场
        plt.subplot(2, 3, 3)
        valid_pois = [poi for poi in poi_queue if poi.result.zncc > 0]
        x = [poi.x for poi in valid_pois]
        y = [poi.y for poi in valid_pois]
        u = [poi.deformation.u for poi in valid_pois]
        v = [poi.deformation.v for poi in valid_pois]
        
        plt.quiver(x, y, u, v, scale=50, width=0.002)
        plt.imshow(ref_cv_img, cmap='gray', alpha=0.5)
        plt.title("Displacement Field")
        plt.axis('off')
        
        # 绘制ZNCC分布
        plt.subplot(2, 3, 4)
        zncc_values = [poi.result.zncc for poi in valid_pois]
        plt.scatter(x, y, c=zncc_values, cmap='jet', s=10)
        plt.colorbar(label='ZNCC')
        plt.title("ZNCC Distribution")
        plt.axis('off')
        
        # 绘制x方向变形梯度
        plt.subplot(2, 3, 5)
        ux_values = [poi.deformation.ux for poi in valid_pois]
        plt.scatter(x, y, c=ux_values, cmap='jet', s=10)
        plt.colorbar(label='ux')
        plt.title("X-direction Strain")
        plt.axis('off')
        
        # 绘制y方向变形梯度
        plt.subplot(2, 3, 6)
        vy_values = [poi.deformation.vy for poi in valid_pois]
        plt.scatter(x, y, c=vy_values, cmap='jet', s=10)
        plt.colorbar(label='vy')
        plt.title("Y-direction Strain")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(img_result_path, dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main() 