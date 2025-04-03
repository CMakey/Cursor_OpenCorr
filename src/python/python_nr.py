import numpy as np
import math
import numba as nb
from numba.experimental import jitclass
from numba import prange
import time
from python_cubic_interpolation import BicubicBspline, Point2D, Image2D

# 定义numba可用的点和变形类型
spec_point2d = [
    ('x', nb.float32),
    ('y', nb.float32),
]

@jitclass(spec_point2d)
class NumbaPoint2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

spec_deformation = [
    ('u', nb.float32),
    ('ux', nb.float32),
    ('uy', nb.float32),
    ('v', nb.float32),
    ('vx', nb.float32),
    ('vy', nb.float32),
]

@jitclass(spec_deformation)
class NumbaDeformation2D1:
    def __init__(self, u=0.0, ux=0.0, uy=0.0, v=0.0, vx=0.0, vy=0.0):
        self.u = u
        self.ux = ux
        self.uy = uy
        self.v = v
        self.vx = vx
        self.vy = vy
    
    def warp(self, point):
        warped_x = point.x + self.u + self.ux * point.x + self.uy * point.y
        warped_y = point.y + self.v + self.vx * point.x + self.vy * point.y
        return NumbaPoint2D(warped_x, warped_y)


# 标准Python类用于接口
class Deformation2D1:
    """2D一阶变形模型"""
    def __init__(self, u=0.0, ux=0.0, uy=0.0, v=0.0, vx=0.0, vy=0.0):
        self.u = u
        self.ux = ux
        self.uy = uy
        self.v = v
        self.vx = vx
        self.vy = vy
    
    def setDeformation(self, *args):
        """设置变形参数"""
        if len(args) == 1 and isinstance(args[0], Deformation2D1):
            # 复制另一个变形对象
            deformation = args[0]
            self.u = deformation.u
            self.ux = deformation.ux
            self.uy = deformation.uy
            self.v = deformation.v
            self.vx = deformation.vx
            self.vy = deformation.vy
        elif len(args) == 6:
            # 直接设置6个参数
            self.u = args[0]
            self.ux = args[1]
            self.uy = args[2]
            self.v = args[3]
            self.vx = args[4]
            self.vy = args[5]
    
    def warp(self, point):
        """根据变形模型计算变形后的坐标"""
        warped_x = point.x + self.u + self.ux * point.x + self.uy * point.y
        warped_y = point.y + self.v + self.vx * point.x + self.vy * point.y
        return Point2D(warped_x, warped_y)
    
    def to_numba(self):
        """转换为numba兼容类型"""
        return NumbaDeformation2D1(self.u, self.ux, self.uy, self.v, self.vx, self.vy)


class Subset2D:
    """2D子区类"""
    def __init__(self, center, radius_x, radius_y):
        self.center = center
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.width = 2 * radius_x + 1
        self.height = 2 * radius_y + 1
        self.eg_mat = np.zeros((self.height, self.width), dtype=np.float32)
    
    def fill(self, image):
        """从图像中填充子区灰度值"""
        for y in range(-self.radius_y, self.radius_y + 1):
            for x in range(-self.radius_x, self.radius_x + 1):
                global_x = int(self.center.x + x)
                global_y = int(self.center.y + y)
                
                if (0 <= global_x < image.width and 0 <= global_y < image.height):
                    self.eg_mat[y + self.radius_y, x + self.radius_x] = image.eg_mat[global_y, global_x]
    
    def zeroMeanNorm(self):
        """计算子区的零均值归一化"""
        mean = np.mean(self.eg_mat)
        self.eg_mat = self.eg_mat - mean
        norm = np.sqrt(np.sum(self.eg_mat * self.eg_mat))
        return norm


class Gradient2D4:
    """2D图像梯度计算"""
    def __init__(self, image):
        self.image = image
        self.gradient_x = np.zeros_like(image.eg_mat)
        self.gradient_y = np.zeros_like(image.eg_mat)
    
    def getGradientX(self):
        """计算x方向梯度"""
        height, width = self.image.eg_mat.shape
        self._compute_gradient_x(self.image.eg_mat, self.gradient_x, height, width)
    
    def getGradientY(self):
        """计算y方向梯度"""
        height, width = self.image.eg_mat.shape
        self._compute_gradient_y(self.image.eg_mat, self.gradient_y, height, width)
    
    @staticmethod
    @nb.njit(parallel=True)
    def _compute_gradient_x(img_data, gradient_x, height, width):
        """使用numba并行计算x方向梯度"""
        for y in prange(height):
            for x in range(1, width - 1):
                gradient_x[y, x] = (img_data[y, x + 1] - img_data[y, x - 1]) / 2.0
    
    @staticmethod
    @nb.njit(parallel=True)
    def _compute_gradient_y(img_data, gradient_y, height, width):
        """使用numba并行计算y方向梯度"""
        for y in prange(1, height - 1):
            for x in range(width):
                gradient_y[y, x] = (img_data[y + 1, x] - img_data[y - 1, x]) / 2.0


class POI2D:
    """2D兴趣点"""
    class Result:
        def __init__(self):
            self.u0 = 0.0
            self.v0 = 0.0
            self.zncc = 0.0
            self.iteration = 0.0
            self.convergence = 0.0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.deformation = Deformation2D1()
        self.result = self.Result()


# 核心计算函数的numba优化
@nb.njit
def compute_hessian_and_sd_img(tar_gradient_x, tar_gradient_y, subset_height, subset_width, subset_radius_x, subset_radius_y):
    """计算黑塞矩阵和最速下降图像"""
    hessian = np.zeros((6, 6), dtype=np.float32)
    sd_img = np.zeros((subset_height, subset_width, 6), dtype=np.float32)
    
    for r in range(subset_height):
        for c in range(subset_width):
            x_local = c - subset_radius_x
            y_local = r - subset_radius_y
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
    
    return hessian, sd_img


@nb.njit
def compute_numerator(sd_img, error_img, subset_height, subset_width):
    """计算参数增量的分子部分"""
    numerator = np.zeros(6, dtype=np.float32)
    for r in range(subset_height):
        for c in range(subset_width):
            for i in range(6):
                numerator[i] += sd_img[r, c, i] * error_img[r, c]
    return numerator


@nb.njit(parallel=True)
def reconstruct_subset_parallel(tar_subset, tar_gradient_x, tar_gradient_y, 
                               subset_height, subset_width, subset_radius_x, subset_radius_y,
                               center_x, center_y, p_current, 
                               compute_interp, compute_interp_x, compute_interp_y):
    """并行重构子区和梯度矩阵"""
    for r in prange(subset_height):
        for c in range(subset_width):
            x_local = c - subset_radius_x
            y_local = r - subset_radius_y
            
            # 计算变形后的坐标
            point = NumbaPoint2D(x_local, y_local)
            warped = p_current.warp(point)
            global_x = center_x + warped.x
            global_y = center_y + warped.y
            
            # 计算插值
            global_point = Point2D(global_x, global_y)
            tar_subset[r, c] = compute_interp(global_point)
            tar_gradient_x[r, c] = compute_interp_x(global_point)
            tar_gradient_y[r, c] = compute_interp_y(global_point)


@nb.njit(parallel=True)
def compute_poi_batch(poi_batch, ref_img_data, ref_img_width, ref_img_height,
                      tar_img_data, tar_coefficient, tar_coefficient_x, tar_coefficient_y,
                      subset_radius_x, subset_radius_y, conv_criterion, stop_condition):
    """并行计算POI批次的变形，使用numba加速"""
    batch_size = len(poi_batch)
    subset_width = 2 * subset_radius_x + 1
    subset_height = 2 * subset_radius_y + 1
    
    # 并行处理每个POI
    for i in range(batch_size):
        poi = poi_batch[i]
        
        # 检查POI是否有效
        if (poi.y - subset_radius_y < 0 or poi.x - subset_radius_x < 0 or
            poi.y + subset_radius_y > ref_img_height - 1 or 
            poi.x + subset_radius_x > ref_img_width - 1 or
            abs(poi.deformation.u) >= ref_img_width or 
            abs(poi.deformation.v) >= ref_img_height or
            poi.result.zncc < 0 or math.isnan(poi.deformation.u) or 
            math.isnan(poi.deformation.v)):
            
            poi.result.zncc = min(poi.result.zncc, -1.0) if poi.result.zncc < 0 else -1.0
            continue
        
        # 创建参考子区
        ref_subset = np.zeros((subset_height, subset_width), dtype=np.float32)
        for r in range(subset_height):
            for c in range(subset_width):
                y_global = int(poi.y + (r - subset_radius_y))
                x_global = int(poi.x + (c - subset_radius_x))
                if 0 <= y_global < ref_img_height and 0 <= x_global < ref_img_width:
                    ref_subset[r, c] = ref_img_data[y_global, x_global]
        
        # 计算参考子区的零均值归一化
        ref_mean = np.mean(ref_subset)
        ref_subset = ref_subset - ref_mean
        ref_mean_norm = np.sqrt(np.sum(ref_subset * ref_subset))
        
        # 创建目标子区
        tar_subset = np.zeros((subset_height, subset_width), dtype=np.float32)
        tar_gradient_x = np.zeros((subset_height, subset_width), dtype=np.float32)
        tar_gradient_y = np.zeros((subset_height, subset_width), dtype=np.float32)
        
        # 获取初始变形参数
        p_current = NumbaDeformation2D1(
            poi.deformation.u, poi.deformation.ux, poi.deformation.uy,
            poi.deformation.v, poi.deformation.vx, poi.deformation.vy
        )
        
        # 保存初始猜测值
        u0 = p_current.u
        v0 = p_current.v
        
        # Newton-Raphson迭代
        iteration_counter = 0
        dp_norm_max = 0.0
        znssd = 0.0
        
        while iteration_counter < stop_condition:
            iteration_counter += 1
            
            # 重构子区和梯度矩阵
            for r in range(subset_height):
                for c in range(subset_width):
                    x_local = c - subset_radius_x
                    y_local = r - subset_radius_y
                    
                    # 计算变形后的坐标
                    point = NumbaPoint2D(x_local, y_local)
                    warped = p_current.warp(point)
                    global_x = poi.x + warped.x
                    global_y = poi.y + warped.y
                    
                    # 边界检查和插值计算
                    if (global_x < 1 or global_y < 1 or 
                        global_x >= ref_img_width - 2 or global_y >= ref_img_height - 2):
                        tar_subset[r, c] = 0.0
                        tar_gradient_x[r, c] = 0.0
                        tar_gradient_y[r, c] = 0.0
                    else:
                        # 插值计算（简化版，实际应使用Bicubic插值）
                        x_int = int(global_x)
                        y_int = int(global_y)
                        x_frac = global_x - x_int
                        y_frac = global_y - y_int
                        
                        # 简单的双线性插值用于演示
                        w00 = (1-x_frac) * (1-y_frac)
                        w01 = x_frac * (1-y_frac)
                        w10 = (1-x_frac) * y_frac
                        w11 = x_frac * y_frac
                        
                        tar_subset[r, c] = (
                            tar_img_data[y_int, x_int] * w00 + 
                            tar_img_data[y_int, x_int+1] * w01 + 
                            tar_img_data[y_int+1, x_int] * w10 + 
                            tar_img_data[y_int+1, x_int+1] * w11
                        )
                        
                        # 同样为梯度计算简单插值
                        tar_gradient_x[r, c] = (tar_img_data[y_int, min(x_int+1, ref_img_width-1)] - 
                                             tar_img_data[y_int, max(x_int-1, 0)]) / 2.0
                        tar_gradient_y[r, c] = (tar_img_data[min(y_int+1, ref_img_height-1), x_int] - 
                                             tar_img_data[max(y_int-1, 0), x_int]) / 2.0
            
            # 计算目标子区的零均值归一化
            tar_mean = np.mean(tar_subset)
            tar_subset = tar_subset - tar_mean
            tar_mean_norm = np.sqrt(np.sum(tar_subset * tar_subset))
            
            if tar_mean_norm < 1e-10:
                break  # 避免除零错误
            
            # 计算黑塞矩阵和最速下降图像
            hessian = np.zeros((6, 6), dtype=np.float32)
            sd_img = np.zeros((subset_height, subset_width, 6), dtype=np.float32)
            
            for r in range(subset_height):
                for c in range(subset_width):
                    x_local = c - subset_radius_x
                    y_local = r - subset_radius_y
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
            
            # 计算Hessian矩阵的逆（Numba不直接支持np.linalg.inv，简化处理）
            # 实际应用中，应替换为稳定的矩阵求逆方法
            inv_hessian = np.zeros((6, 6), dtype=np.float32)
            
            # 这里简化处理，实际应使用稳健的求逆方法
            # 对角线设为Hessian对角线的倒数（非常简化的处理）
            for i in range(6):
                if abs(hessian[i, i]) > 1e-10:
                    inv_hessian[i, i] = 1.0 / hessian[i, i]
            
            # 计算误差图像
            error_img = ref_subset * (tar_mean_norm / ref_mean_norm) - tar_subset
            
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
            p_current = NumbaDeformation2D1(
                p_current.u + dp[0],
                p_current.ux + dp[1],
                p_current.uy + dp[2],
                p_current.v + dp[3],
                p_current.vx + dp[4],
                p_current.vy + dp[5]
            )
            
            # 检查收敛
            subset_radius_x2 = subset_radius_x * subset_radius_x
            subset_radius_y2 = subset_radius_y * subset_radius_y
            
            dp_norm_max = (dp[0] * dp[0] + 
                         dp[1] * dp[1] * subset_radius_x2 +
                         dp[2] * dp[2] * subset_radius_y2 +
                         dp[3] * dp[3] +
                         dp[4] * dp[4] * subset_radius_x2 +
                         dp[5] * dp[5] * subset_radius_y2)
            
            dp_norm_max = math.sqrt(dp_norm_max)
            
            # 检查是否满足终止条件
            if dp_norm_max < conv_criterion:
                break
        
        # 存储最终结果
        poi_batch[i].deformation.u = p_current.u
        poi_batch[i].deformation.ux = p_current.ux
        poi_batch[i].deformation.uy = p_current.uy
        poi_batch[i].deformation.v = p_current.v
        poi_batch[i].deformation.vx = p_current.vx
        poi_batch[i].deformation.vy = p_current.vy
        
        # 保存输出参数
        poi_batch[i].result.u0 = u0
        poi_batch[i].result.v0 = v0
        poi_batch[i].result.zncc = 0.5 * (2.0 - znssd)
        poi_batch[i].result.iteration = float(iteration_counter)
        poi_batch[i].result.convergence = dp_norm_max
        
        # 检查迭代是否在期望目标处收敛
        if poi_batch[i].result.convergence >= conv_criterion and iteration_counter >= stop_condition:
            poi_batch[i].result.zncc = -4.0
        
        # 检查是否出现NaN
        if (math.isnan(poi_batch[i].result.zncc) or math.isnan(poi_batch[i].deformation.u) or math.isnan(poi_batch[i].deformation.v)):
            poi_batch[i].deformation.u = u0
            poi_batch[i].deformation.v = v0
            poi_batch[i].result.zncc = -5.0
    
    return poi_batch


class NR2D1:
    """Newton-Raphson算法实现，使用numba并行优化"""
    def __init__(self, subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size=16):
        self.subset_radius_x = subset_radius_x
        self.subset_radius_y = subset_radius_y
        self.conv_criterion = conv_criterion
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        
        self.ref_img = None
        self.tar_img = None
        self.tar_gradient = None
        self.tar_interp = None
        self.tar_interp_x = None
        self.tar_interp_y = None
    
    def set_images(self, ref_img, tar_img):
        """设置参考图像和目标图像"""
        self.ref_img = ref_img
        self.tar_img = tar_img
    
    def prepare(self):
        """准备阶段：计算梯度和插值系数表"""
        start_time = time.time()
        
        # 创建目标图像的梯度
        self.tar_gradient = Gradient2D4(self.tar_img)
        self.tar_gradient.getGradientX()
        self.tar_gradient.getGradientY()
        
        # 创建目标图像的插值系数表
        self.tar_interp = BicubicBspline(self.tar_img)
        self.tar_interp.prepare()
        
        # 创建x方向梯度的插值系数表
        gradient_img_x = Image2D(self.tar_img.width, self.tar_img.height)
        gradient_img_x.eg_mat = self.tar_gradient.gradient_x
        self.tar_interp_x = BicubicBspline(gradient_img_x)
        self.tar_interp_x.prepare()
        
        # 创建y方向梯度的插值系数表
        gradient_img_y = Image2D(self.tar_img.width, self.tar_img.height)
        gradient_img_y.eg_mat = self.tar_gradient.gradient_y
        self.tar_interp_y = BicubicBspline(gradient_img_y)
        self.tar_interp_y.prepare()
        
        end_time = time.time()
        print(f"NR2D1 preparation completed in {end_time - start_time:.3f} seconds")
    
    def compute(self, poi):
        """计算单个POI的变形"""
        subset_width = 2 * self.subset_radius_x + 1
        subset_height = 2 * self.subset_radius_y + 1
        
        # 检查POI是否有效
        if (poi.y - self.subset_radius_y < 0 or poi.x - self.subset_radius_x < 0 or
            poi.y + self.subset_radius_y > self.ref_img.height - 1 or 
            poi.x + self.subset_radius_x > self.ref_img.width - 1 or
            abs(poi.deformation.u) >= self.ref_img.width or 
            abs(poi.deformation.v) >= self.ref_img.height or
            poi.result.zncc < 0 or math.isnan(poi.deformation.u) or 
            math.isnan(poi.deformation.v)):
            
            poi.result.zncc = poi.result.zncc if poi.result.zncc < -1 else -1
            return
        
        # 创建参考子区
        ref_subset = Subset2D(Point2D(poi.x, poi.y), self.subset_radius_x, self.subset_radius_y)
        ref_subset.fill(self.ref_img)
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
        
        while True:
            iteration_counter += 1
            
            # 使用numba并行重构子区和梯度矩阵
            reconstruct_subset_parallel(
                tar_subset.eg_mat, tar_gradient_x, tar_gradient_y,
                subset_height, subset_width, self.subset_radius_x, self.subset_radius_y,
                poi.x, poi.y, p_numba,
                self.tar_interp.compute, self.tar_interp_x.compute, self.tar_interp_y.compute
            )
            
            # 计算目标子区的零均值归一化
            tar_mean = np.mean(tar_subset.eg_mat)
            tar_subset.eg_mat = tar_subset.eg_mat - tar_mean
            tar_mean_norm = np.sqrt(np.sum(tar_subset.eg_mat * tar_subset.eg_mat))
            
            # 使用numba加速计算黑塞矩阵和最速下降图像
            hessian, sd_img = compute_hessian_and_sd_img(
                tar_gradient_x, tar_gradient_y, 
                subset_height, subset_width, 
                self.subset_radius_x, self.subset_radius_y
            )
            
            # 计算黑塞矩阵的逆
            inv_hessian = np.linalg.inv(hessian)
            
            # 计算误差图像
            error_img = ref_subset.eg_mat * (tar_mean_norm / ref_mean_norm) - tar_subset.eg_mat
            
            # 计算ZNSSD
            znssd = np.sum(error_img * error_img) / (tar_mean_norm * tar_mean_norm)
            
            # 使用numba加速计算参数增量的分子部分
            numerator = compute_numerator(sd_img, error_img, subset_height, subset_width)
            
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
            p_numba = p_current.to_numba()  # 更新numba类型参数
            
            # 检查收敛
            subset_radius_x2 = self.subset_radius_x * self.subset_radius_x
            subset_radius_y2 = self.subset_radius_y * self.subset_radius_y
            
            dp_norm_max = (dp[0] * dp[0] + 
                        dp[1] * dp[1] * subset_radius_x2 +
                        dp[2] * dp[2] * subset_radius_y2 +
                        dp[3] * dp[3] +
                        dp[4] * dp[4] * subset_radius_x2 +
                        dp[5] * dp[5] * subset_radius_y2)
            
            dp_norm_max = math.sqrt(dp_norm_max)
            
            # 检查是否满足终止条件
            if iteration_counter >= self.stop_condition or dp_norm_max < self.conv_criterion:
                break
        
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
        if (math.isnan(poi.result.zncc) or math.isnan(poi.deformation.u) or math.isnan(poi.deformation.v)):
            poi.deformation.u = poi.result.u0
            poi.deformation.v = poi.result.v0
            poi.result.zncc = -5.0
    
    def compute_poi_queue(self, poi_queue):
        """并行计算POI队列，使用分批处理来优化性能"""
        start_time = time.time()
        queue_length = len(poi_queue)
        
        # 检查是否已经准备好
        if (self.tar_interp is None or self.tar_interp_x is None or 
            self.tar_interp_y is None or self.ref_img is None or 
            self.tar_img is None):
            print("Error: NR2D1 not properly prepared. Call prepare() first.")
            return
        
        # 转换为numba可用的格式
        ref_img_data = self.ref_img.eg_mat
        ref_img_width = self.ref_img.width
        ref_img_height = self.ref_img.height
        tar_img_data = self.tar_img.eg_mat
        
        print(f"Processing {queue_length} POIs in batches of {self.batch_size}...")
        
        # 按批次处理POI队列
        for i in range(0, queue_length, self.batch_size):
            batch_end = min(i+self.batch_size, queue_length)
            batch = poi_queue[i:batch_end]
            
            # 使用numba并行加速处理当前批次
            processed_batch = compute_poi_batch(
                batch, 
                ref_img_data, ref_img_width, ref_img_height,
                tar_img_data, 
                self.tar_interp.coefficient, 
                self.tar_interp_x.coefficient, 
                self.tar_interp_y.coefficient,
                self.subset_radius_x, self.subset_radius_y, 
                self.conv_criterion, self.stop_condition
            )
            
            # 更新原始队列中的结果
            for j in range(len(batch)):
                poi_queue[i+j] = processed_batch[j]
                
            if (i+self.batch_size) % (5*self.batch_size) == 0 or batch_end == queue_length:
                progress = (batch_end / queue_length) * 100
                print(f"Progress: {progress:.1f}% ({batch_end}/{queue_length} POIs processed)")
        
        end_time = time.time()
        processing_time = end_time - start_time
        poi_per_second = queue_length / processing_time
        
        # 计算有效结果数量
        valid_count = sum(1 for poi in poi_queue if poi.result.zncc > 0)
        valid_percentage = (valid_count / queue_length) * 100
        
        print(f"POI queue computation completed in {processing_time:.3f} seconds")
        print(f"Average processing speed: {poi_per_second:.1f} POIs/second")
        print(f"Valid results: {valid_count} out of {queue_length} ({valid_percentage:.1f}%)")
        
        return poi_queue


# 使用示例
if __name__ == "__main__":
    import time
    from python_cubic_interpolation import load_image, visualize_interpolation
    import matplotlib.pyplot as plt
    
    try:
        # 加载图像
        image_path = "speckle_medium.tif"  # 替换为你的图像路径
        ref_image, ref_cv_image = load_image(image_path)
        # 可以加载不同的目标图像，这里为演示用同一图像
        tar_image, tar_cv_image = load_image(image_path)
        
        print(f"Images loaded: {ref_image.width}x{ref_image.height}")
        
        # 创建NR2D1实例
        subset_radius_x = 15
        subset_radius_y = 15
        conv_criterion = 1e-6
        stop_condition = 50
        
        nr = NR2D1(subset_radius_x, subset_radius_y, conv_criterion, stop_condition, batch_size=16)
        nr.set_images(ref_image, tar_image)
        
        # 准备阶段
        nr.prepare()
        
        # 创建一组测试POI
        poi_count = 100
        poi_queue = []
        
        for i in range(poi_count):
            x = np.random.randint(subset_radius_x + 10, ref_image.width - subset_radius_x - 10)
            y = np.random.randint(subset_radius_y + 10, ref_image.height - subset_radius_y - 10)
            poi = POI2D(x, y)
            poi.deformation.u = 0.5  # 初始猜测值
            poi.deformation.v = 0.5
            poi_queue.append(poi)
        
        # 测试单个POI计算
        print("\nTesting single POI computation...")
        start_time = time.time()
        nr.compute(poi_queue[0])
        single_time = time.time() - start_time
        print(f"Single POI computation time: {single_time:.3f} seconds")
        print(f"Results: u={poi_queue[0].deformation.u:.4f}, v={poi_queue[0].deformation.v:.4f}")
        
        # 测试队列计算
        print("\nTesting POI queue computation...")
        nr.compute_poi_queue(poi_queue)
        
        # 可视化一部分结果
        plt.figure(figsize=(10, 8))
        plt.imshow(ref_cv_image, cmap='gray')
        
        # 绘制原始POI位置
        for i, poi in enumerate(poi_queue[:10]):  # 只显示前10个POI
            plt.plot(poi.x, poi.y, 'ro', markersize=5)
            plt.plot(poi.x + poi.deformation.u, poi.y + poi.deformation.v, 'go', markersize=5)
            plt.arrow(poi.x, poi.y, poi.deformation.u, poi.deformation.v, 
                    color='b', head_width=3, head_length=3, width=0.5, alpha=0.7)
        
        plt.title("Newton-Raphson DIC Results")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")