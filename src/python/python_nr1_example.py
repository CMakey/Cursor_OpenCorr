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
from python_nr import NR2D1, POI2D, Deformation2D1
from python_gradient import Gradient2D4

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
    tar_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/Camera_DEV_1AB22C0222A5_2024-09-23_19-29-40.png"     # 替换为您计算机上的路径
    
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
    subset_radius_x = 16
    subset_radius_y = 16
    max_iteration = 10
    max_deformation_norm = 0.001
    
    # 设置POI
    upper_left_point = Point2D(975, 695)
    poi_number_x = 9  # 减少点数以加快示例运行速度
    poi_number_y = 7
    grid_space = 200
    
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
    # transform_matrix = np.array([
    #     [1.02, 0.01, 2.5],
    #     [0.01, 1.03, 1.5],
    #     [0.0001, 0.0001, 1.0]
    # ])
    # persp_init.setTransformMatrix(transform_matrix)
    transform_matrix = np.array([
        [ 8.25327830e-01, -1.92603161e-02,  6.08026082e+02],
        [ 9.29342396e-03,  8.11131732e-01,  1.28319036e+02], 
        [ 1.38626335e-06, -5.94219881e-06,  1.00000000e+00]
    ])
    persp_init.setTransformMatrix(transform_matrix)
    
    # 创建和准备梯度计算实例 - 使用python_gradient模块而不是内置于NR类
    tar_gradient = Gradient2D4(tar_img.eg_mat)
    tar_gradient_x = tar_gradient.get_gradient_x()
    tar_gradient_y = tar_gradient.get_gradient_y()
    
    # 创建和准备插值实例 - 使用python_cubic_interpolation模块
    tar_interp = BicubicBspline(tar_img)
    tar_interp.prepare()
    
    # 创建梯度x和y方向的插值
    gradient_x_img = Image2D(tar_img.width, tar_img.height, tar_gradient_x)
    tar_interp_x = BicubicBspline(gradient_x_img)
    tar_interp_x.prepare()
    
    gradient_y_img = Image2D(tar_img.width, tar_img.height, tar_gradient_y)
    tar_interp_y = BicubicBspline(gradient_y_img)
    tar_interp_y.prepare()
    
    # 创建NR实例
    nr = NR2D1(subset_radius_x, subset_radius_y, max_deformation_norm, max_iteration, batch_size=16)
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
    # 将准备好的梯度和插值数据设置到NR实例中
    nr.tar_gradient_x = tar_gradient_x
    nr.tar_gradient_y = tar_gradient_y
    nr.tar_coefficient = tar_interp.coefficient
    nr.tar_coefficient_x = tar_interp_x.coefficient
    nr.tar_coefficient_y = tar_interp_y.coefficient
    nr.is_prepared = True  # 表示已经准备好了数据，跳过内部prepare()步骤
    
    nr.compute_poi_queue(poi_queue)
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