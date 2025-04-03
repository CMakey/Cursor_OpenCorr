# bicubic_bspline_parallel.py

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import numba as nb
import time 

class Point2D:
    """2D point class to match the C++ implementation."""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Image2D:
    """Simple 2D image class to match the C++ implementation."""
    def __init__(self, width, height, data=None):
        self.width = width
        self.height = height
        if data is not None:
            self.eg_mat = data
        else:
            self.eg_mat = np.zeros((height, width), dtype=np.float32)


# 使用numba优化的并行计算函数
@nb.njit(parallel=True)
def calculate_coefficients(height, width, img_data, bc_matrix):
    """Parallel coefficient calculation using numba."""
    coefficient = np.zeros((height, width, 4, 4), dtype=np.float32)
    
    for r in nb.prange(1, height - 2):  # parallel for
        for c in range(1, width - 2):
            # 填充4x4网格的灰度值
            mat_q = np.zeros((4, 4), dtype=np.float32)
            for i in range(4):
                for j in range(4):
                    mat_q[i, j] = img_data[r - 1 + i, c - 1 + j]
            
            # 计算插值系数矩阵
            mat_p = np.zeros((4, 4), dtype=np.float32)
            for k in range(4):
                for l in range(4):
                    for m in range(4):
                        for n in range(4):
                            mat_p[k, l] += bc_matrix[l, m] * bc_matrix[k, n] * mat_q[n, m]
            
            # 重排系数
            for k in range(4):
                for l in range(4):
                    coefficient[r, c, k, l] = mat_p[3 - k, 3 - l]
    
    return coefficient


class BicubicBspline:
    """
    Python implementation of BicubicBspline interpolation from OpenCorr with parallel processing.
    Based on Z. Pan et al, Theoretical and Applied Mechanics Letters (2016) 6(3): 126-130.
    """
    
    def __init__(self, image):
        """Initialize with an image."""
        self.coefficient = None
        self.interp_img = None
        self.height = 0
        self.width = 0
        self.set_image(image)
        
        # BC = B * C (预计算矩阵)
        self.BC_MATRIX = np.array([
            [-144.0/336.0, 384.0/336.0, -384.0/336.0, 144.0/336.0],
            [342.0/336.0, -702.0/336.0, 450.0/336.0, -90.0/336.0],
            [-198.0/336.0, -18.0/336.0, 270.0/336.0, -54.0/336.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
    
    def set_image(self, image):
        """Set the image to process."""
        if image.height < 5 or image.width < 5:
            print(f"Too small image: {image.width}, {image.height}")
        else:
            self.interp_img = image
            self.height = image.height
            self.width = image.width
    
    def prepare(self):
        """Prepare interpolation coefficients using parallel processing."""
        start_time = time.time()
        
        self.coefficient = calculate_coefficients(
            self.height,
            self.width,
            self.interp_img.eg_mat,
            self.BC_MATRIX
        )
        
        end_time = time.time()
        print(f"[prepare] Interpolation coefficient preparation time: {end_time - start_time:.4f} seconds")
    
    def compute(self, location):
        """Compute interpolated value at given location."""
        value = 0.0
        
        # Check boundary conditions
        if (location.x < 1 or location.y < 1 or 
            location.x >= self.width - 2 or location.y >= self.height - 2 or 
            math.isnan(location.x) or math.isnan(location.y)):
            value = -1.0
        else:
            # Split coordinates into integral and decimal parts
            x_integral = math.floor(location.x)
            y_integral = math.floor(location.y)
            
            x_decimal = location.x - x_integral
            y_decimal = location.y - y_integral
            
            x2_decimal = x_decimal * x_decimal
            y2_decimal = y_decimal * y_decimal
            
            x3_decimal = x2_decimal * x_decimal
            y3_decimal = y2_decimal * y_decimal
            
            # Get local coefficient matrix
            local_coefficient = self.coefficient[y_integral, x_integral]
            
            # Compute interpolated value using the bicubic polynomial
            value = local_coefficient[0, 0] \
                + local_coefficient[0, 1] * x_decimal \
                + local_coefficient[0, 2] * x2_decimal \
                + local_coefficient[0, 3] * x3_decimal \
                \
                + local_coefficient[1, 0] * y_decimal \
                + local_coefficient[1, 1] * y_decimal * x_decimal \
                + local_coefficient[1, 2] * y_decimal * x2_decimal \
                + local_coefficient[1, 3] * y_decimal * x3_decimal \
                \
                + local_coefficient[2, 0] * y2_decimal \
                + local_coefficient[2, 1] * y2_decimal * x_decimal \
                + local_coefficient[2, 2] * y2_decimal * x2_decimal \
                + local_coefficient[2, 3] * y2_decimal * x3_decimal \
                \
                + local_coefficient[3, 0] * y3_decimal \
                + local_coefficient[3, 1] * y3_decimal * x_decimal \
                + local_coefficient[3, 2] * y3_decimal * x2_decimal \
                + local_coefficient[3, 3] * y3_decimal * x3_decimal
            
        return value


# 并行处理图像上采样
@nb.njit(parallel=True)
def parallel_upscale(src_width, src_height, new_width, new_height, scale_factor, coefficient):
    """Parallel image upscaling using numba."""
    upscaled_image = np.zeros((new_height, new_width), dtype=np.float32)
    
    for y in nb.prange(new_height):  # parallel for
        for x in range(new_width):
            # 映射到原始图像坐标
            orig_x = x / scale_factor
            orig_y = y / scale_factor
            
            # 检查边界条件
            if (orig_x < 1 or orig_y < 1 or 
                orig_x >= src_width - 2 or orig_y >= src_height - 2):
                # 边界情况使用最近邻插值
                orig_x_int = min(max(int(orig_x), 0), src_width - 1)
                orig_y_int = min(max(int(orig_y), 0), src_height - 1)
                upscaled_image[y, x] = -1.0  # 标记为边界值，稍后处理
            else:
                # 计算整数部分和小数部分
                x_integral = int(orig_x)
                y_integral = int(orig_y)
                
                x_decimal = orig_x - x_integral
                y_decimal = orig_y - y_integral
                
                x2_decimal = x_decimal * x_decimal
                y2_decimal = y_decimal * y_decimal
                
                x3_decimal = x2_decimal * x_decimal
                y3_decimal = y2_decimal * y_decimal
                
                # 获取局部系数
                local_coefficient = coefficient[y_integral, x_integral]
                
                # 计算插值值
                value = local_coefficient[0, 0] \
                    + local_coefficient[0, 1] * x_decimal \
                    + local_coefficient[0, 2] * x2_decimal \
                    + local_coefficient[0, 3] * x3_decimal \
                    \
                    + local_coefficient[1, 0] * y_decimal \
                    + local_coefficient[1, 1] * y_decimal * x_decimal \
                    + local_coefficient[1, 2] * y_decimal * x2_decimal \
                    + local_coefficient[1, 3] * y_decimal * x3_decimal \
                    \
                    + local_coefficient[2, 0] * y2_decimal \
                    + local_coefficient[2, 1] * y2_decimal * x_decimal \
                    + local_coefficient[2, 2] * y2_decimal * x2_decimal \
                    + local_coefficient[2, 3] * y2_decimal * x3_decimal \
                    \
                    + local_coefficient[3, 0] * y3_decimal \
                    + local_coefficient[3, 1] * y3_decimal * x_decimal \
                    + local_coefficient[3, 2] * y3_decimal * x2_decimal \
                    + local_coefficient[3, 3] * y3_decimal * x3_decimal
                
                upscaled_image[y, x] = value
                
    return upscaled_image


def load_image(image_path):
    """Load an image using OpenCV and convert it to our format."""
    # Read image
    cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if cv_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get dimensions
    height, width = cv_image.shape
    
    # Convert to float32
    cv_image_float = cv_image.astype(np.float32)
    
    # Create our image object
    image = Image2D(width, height, cv_image_float)
    
    return image, cv_image


def visualize_interpolation(original_image, interpolated_image, title="Interpolation Comparison"):
    """Visualize original and interpolated images."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(interpolated_image, cmap='gray')
    plt.title("Interpolated Image")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# 示例使用
if __name__ == "__main__":
    import time
    
    try:
        # 加载图像
        image_path = "speckle_medium.tif"  # 替换为您的图像路径
        image, original_cv_image = load_image(image_path)
        
        print(f"Image loaded: {image.width}x{image.height}")
        
        # 创建并准备双三次B样条插值器
        start_time = time.time()
        interpolator = BicubicBspline(image)
        interpolator.prepare()
        prepare_time = time.time() - start_time
        print(f"Preparation completed in {prepare_time:.3f} seconds")
        
        # 使用并行处理进行图像上采样
        scale_factor = 1.5
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        start_time = time.time()
        upscaled_image = parallel_upscale(
            image.width, image.height, 
            new_width, new_height, 
            scale_factor, 
            interpolator.coefficient
        )
        
        # 处理边界值（标记为-1的像素）
        boundary_mask = upscaled_image < 0
        if np.any(boundary_mask):
            # 对于边界区域，使用OpenCV进行简单插值
            temp_img = cv2.resize(original_cv_image, (new_width, new_height), 
                                  interpolation=cv2.INTER_LINEAR)
            upscaled_image[boundary_mask] = temp_img[boundary_mask]
        
        upscale_time = time.time() - start_time
        print(f"Upscaling completed in {upscale_time:.3f} seconds")
        
        # 可视化结果
        visualize_interpolation(original_cv_image, upscaled_image, 
                               f"Parallel Bicubic B-spline Interpolation (Scale {scale_factor}x)")
        
        # 与OpenCV的内置插值比较
        start_time = time.time()
        cv_upscaled = cv2.resize(original_cv_image, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
        cv_time = time.time() - start_time
        print(f"OpenCV resizing completed in {cv_time:.3f} seconds")
        
        visualize_interpolation(upscaled_image, cv_upscaled, 
                               "Our Parallel Bicubic B-spline vs OpenCV Bicubic")
        
        # 打印性能对比
        print("\nPerformance Comparison:")
        print(f"Our implementation (parallel): {upscale_time:.3f} seconds")
        print(f"OpenCV implementation: {cv_time:.3f} seconds")
        print(f"Speed ratio: {cv_time/upscale_time:.2f}x")
        
    except Exception as e:
        print(f"Error: {e}")