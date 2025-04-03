"""
This file is part of OpenCorr, an open source Python library for
study and development of 2D, 3D/stereo and volumetric
digital image correlation.

Based on the original C++ implementation in OpenCorr.
"""

import numpy as np
from numba import jit, prange

# 有限差分的权重系数
FIRST_FACTOR = 1.0 / 12.0
SECOND_FACTOR = 2.0 / 3.0

# 创建3D数组的辅助函数
def new3D(dim_z, dim_y, dim_x):
    return np.zeros((dim_z, dim_y, dim_x), dtype=np.float32)


class Gradient2D4:
    """
    用于计算2D图像梯度的类，使用四阶精度的一阶导数方法
    
    基于论文:
    B. Fornberg, Mathematics of Computation (1988) 184(51): 699-706.
    https://doi.org/10.1090/S0025-5718-1988-0935077-0
    """
    
    def __init__(self, image):
        """
        初始化Gradient2D4对象
        
        参数:
            image: 2D numpy数组，表示输入图像
        """
        self.grad_img = image
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_xy = None
    
    def set_image(self, image):
        """设置要处理的图像"""
        self.grad_img = image
    
    def get_gradient_x(self):
        """计算x方向的梯度"""
        height, width = self.grad_img.shape
        self.gradient_x = np.zeros((height, width), dtype=np.float32)
        
        # 调用numba加速的函数
        self._compute_gradient_x(self.grad_img, self.gradient_x, height, width)
        
        return self.gradient_x
    
    def get_gradient_y(self):
        """计算y方向的梯度"""
        height, width = self.grad_img.shape
        self.gradient_y = np.zeros((height, width), dtype=np.float32)
        
        # 调用numba加速的函数
        self._compute_gradient_y(self.grad_img, self.gradient_y, height, width)
        
        return self.gradient_y
    
    def get_gradient_xy(self):
        """计算xy方向的混合偏导数"""
        height, width = self.grad_img.shape
        self.gradient_xy = np.zeros((height, width), dtype=np.float32)
        
        # 如果x方向梯度尚未计算，则先计算
        if self.gradient_x is None or self.gradient_x.shape != (height, width):
            self.get_gradient_x()
        
        # 调用numba加速的函数
        self._compute_gradient_xy(self.gradient_x, self.gradient_xy, height, width)
        
        return self.gradient_xy
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_x(image, gradient, height, width):
        """使用numba加速的x方向梯度计算"""
        for r in prange(height):
            for c in range(2, width - 2):
                result = 0.0
                result -= image[r, c + 2] * FIRST_FACTOR
                result += image[r, c + 1] * SECOND_FACTOR
                result -= image[r, c - 1] * SECOND_FACTOR
                result += image[r, c - 2] * FIRST_FACTOR
                gradient[r, c] = result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_y(image, gradient, height, width):
        """使用numba加速的y方向梯度计算"""
        for r in prange(2, height - 2):
            for c in range(width):
                result = 0.0
                result -= image[r + 2, c] * FIRST_FACTOR
                result += image[r + 1, c] * SECOND_FACTOR
                result -= image[r - 1, c] * SECOND_FACTOR
                result += image[r - 2, c] * FIRST_FACTOR
                gradient[r, c] = result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_xy(gradient_x, gradient_xy, height, width):
        """使用numba加速的xy方向梯度计算"""
        for r in prange(2, height - 2):
            for c in range(width):
                result = 0.0
                result -= gradient_x[r + 2, c] * FIRST_FACTOR
                result += gradient_x[r + 1, c] * SECOND_FACTOR
                result -= gradient_x[r - 1, c] * SECOND_FACTOR
                result += gradient_x[r - 2, c] * FIRST_FACTOR
                gradient_xy[r, c] = result


class Gradient3D4:
    """
    用于计算3D体积图像梯度的类，使用四阶精度的一阶导数方法
    
    基于论文:
    B. Fornberg, Mathematics of Computation (1988) 184(51): 699-706.
    https://doi.org/10.1090/S0025-5718-1988-0935077-0
    """
    
    def __init__(self, image):
        """
        初始化Gradient3D4对象
        
        参数:
            image: 3D numpy数组，表示输入体积图像
        """
        self.grad_img = image
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_z = None
    
    def clear(self):
        """清除所有梯度数据"""
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_z = None
    
    def set_image(self, image):
        """设置要处理的体积图像"""
        self.grad_img = image
    
    def get_gradient_x(self):
        """计算x方向的梯度"""
        dim_z, dim_y, dim_x = self.grad_img.shape
        self.gradient_x = new3D(dim_z, dim_y, dim_x)
        
        # 调用numba加速的函数
        self._compute_gradient_x(self.grad_img, self.gradient_x, dim_z, dim_y, dim_x)
        
        return self.gradient_x
    
    def get_gradient_y(self):
        """计算y方向的梯度"""
        dim_z, dim_y, dim_x = self.grad_img.shape
        self.gradient_y = new3D(dim_z, dim_y, dim_x)
        
        # 调用numba加速的函数
        self._compute_gradient_y(self.grad_img, self.gradient_y, dim_z, dim_y, dim_x)
        
        return self.gradient_y
    
    def get_gradient_z(self):
        """计算z方向的梯度"""
        dim_z, dim_y, dim_x = self.grad_img.shape
        self.gradient_z = new3D(dim_z, dim_y, dim_x)
        
        # 调用numba加速的函数
        self._compute_gradient_z(self.grad_img, self.gradient_z, dim_z, dim_y, dim_x)
        
        return self.gradient_z
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_x(vol_mat, gradient, dim_z, dim_y, dim_x):
        """使用numba加速的x方向梯度计算"""
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(2, dim_x - 2):
                    result = 0.0
                    result -= vol_mat[i, j, k + 2] * FIRST_FACTOR
                    result += vol_mat[i, j, k + 1] * SECOND_FACTOR
                    result -= vol_mat[i, j, k - 1] * SECOND_FACTOR
                    result += vol_mat[i, j, k - 2] * FIRST_FACTOR
                    gradient[i, j, k] = result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_y(vol_mat, gradient, dim_z, dim_y, dim_x):
        """使用numba加速的y方向梯度计算"""
        for k in prange(dim_x):
            for i in range(dim_z):
                for j in range(2, dim_y - 2):
                    result = 0.0
                    result -= vol_mat[i, j + 2, k] * FIRST_FACTOR
                    result += vol_mat[i, j + 1, k] * SECOND_FACTOR
                    result -= vol_mat[i, j - 1, k] * SECOND_FACTOR
                    result += vol_mat[i, j - 2, k] * FIRST_FACTOR
                    gradient[i, j, k] = result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_gradient_z(vol_mat, gradient, dim_z, dim_y, dim_x):
        """使用numba加速的z方向梯度计算"""
        for j in prange(dim_y):
            for k in range(dim_x):
                for i in range(2, dim_z - 2):
                    result = 0.0
                    result -= vol_mat[i + 2, j, k] * FIRST_FACTOR
                    result += vol_mat[i + 1, j, k] * SECOND_FACTOR
                    result -= vol_mat[i - 1, j, k] * SECOND_FACTOR
                    result += vol_mat[i - 2, j, k] * FIRST_FACTOR
                    gradient[i, j, k] = result


# 使用示例
def demo_2d():
    """2D梯度计算示例"""
    # 创建测试图像
    image = np.random.rand(100, 100).astype(np.float32)
    
    # 使用Gradient2D4类计算梯度
    grad = Gradient2D4(image)
    
    # 计算梯度
    grad_x = grad.get_gradient_x()
    grad_y = grad.get_gradient_y()
    grad_xy = grad.get_gradient_xy()
    
    print(f"X梯度形状: {grad_x.shape}")
    print(f"Y梯度形状: {grad_y.shape}")
    print(f"XY梯度形状: {grad_xy.shape}")


def demo_3d():
    """3D梯度计算示例"""
    # 创建测试体积
    volume = np.random.rand(50, 50, 50).astype(np.float32)
    
    # 使用Gradient3D4类计算梯度
    grad = Gradient3D4(volume)
    
    # 计算梯度
    grad_x = grad.get_gradient_x()
    grad_y = grad.get_gradient_y()
    grad_z = grad.get_gradient_z()
    
    print(f"X梯度形状: {grad_x.shape}")
    print(f"Y梯度形状: {grad_y.shape}")
    print(f"Z梯度形状: {grad_z.shape}")


if __name__ == "__main__":
    print("2D梯度计算示例:")
    demo_2d()
    
    print("\n3D梯度计算示例:")
    demo_3d()