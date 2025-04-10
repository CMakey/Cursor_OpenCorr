"""
This file is part of OpenCorr, an open source Python library for
study and development of 2D, 3D/stereo and volumetric
digital image correlation.

Based on the original C++ implementation in OpenCorr.
"""

import numpy as np
import math
from python_cubic_interpolation import Point2D

class Point3D:
    """3D point class to match the C++ implementation."""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class DeformationVector2D:
    """2D一阶变形模型，对应C++中的DeformationVector2D"""
    def __init__(self, u=0.0, ux=0.0, uy=0.0, uxx=0.0, uxy=0.0, uyy=0.0,
                 v=0.0, vx=0.0, vy=0.0, vxx=0.0, vxy=0.0, vyy=0.0):
        self.u = u
        self.ux = ux
        self.uy = uy
        self.uxx = uxx
        self.uxy = uxy
        self.uyy = uyy
        self.v = v
        self.vx = vx
        self.vy = vy
        self.vxx = vxx
        self.vxy = vxy
        self.vyy = vyy
        
    def as_array(self):
        """将变形参数作为数组返回"""
        return np.array([self.u, self.ux, self.uy, self.uxx, self.uxy, self.uyy,
                          self.v, self.vx, self.vy, self.vxx, self.vxy, self.vyy], dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入变形参数"""
        self.u, self.ux, self.uy, self.uxx, self.uxy, self.uyy, \
        self.v, self.vx, self.vy, self.vxx, self.vxy, self.vyy = arr
        
    def warp_first_order(self, point):
        """使用一阶变形模型计算变形后的坐标"""
        warped_x = point.x + self.u + self.ux * point.x + self.uy * point.y
        warped_y = point.y + self.v + self.vx * point.x + self.vy * point.y
        return Point2D(warped_x, warped_y)
    
    def warp_second_order(self, point):
        """使用二阶变形模型计算变形后的坐标"""
        x, y = point.x, point.y
        warped_x = x + self.u + self.ux * x + self.uy * y + \
                   0.5 * self.uxx * x * x + self.uxy * x * y + 0.5 * self.uyy * y * y
        warped_y = y + self.v + self.vx * x + self.vy * y + \
                   0.5 * self.vxx * x * x + self.vxy * x * y + 0.5 * self.vyy * y * y
        return Point2D(warped_x, warped_y)


class StrainVector2D:
    """2D应变模型，对应C++中的StrainVector2D"""
    def __init__(self, exx=0.0, eyy=0.0, exy=0.0):
        self.exx = exx
        self.eyy = eyy
        self.exy = exy
    
    def as_array(self):
        """将应变参数作为数组返回"""
        return np.array([self.exx, self.eyy, self.exy], dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入应变参数"""
        self.exx, self.eyy, self.exy = arr


class Result2D:
    """2D DIC结果，对应C++中的Result2D"""
    def __init__(self, u0=0.0, v0=0.0, zncc=0.0, iteration=0.0, convergence=0.0, feature=0.0):
        self.u0 = u0
        self.v0 = v0
        self.zncc = zncc
        self.znssd = 0.0  # 为兼容性添加，与zncc有关系: zncc = 0.5 * (2 - znssd)
        self.iteration = iteration
        self.convergence = convergence
        self.feature = feature
    
    def as_array(self):
        """将结果参数作为数组返回"""
        return np.array([self.u0, self.v0, self.zncc, self.iteration, self.convergence, self.feature], 
                         dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入结果参数"""
        self.u0, self.v0, self.zncc, self.iteration, self.convergence, self.feature = arr
    
    def update_znssd_from_zncc(self):
        """根据ZNCC更新ZNSSD值"""
        self.znssd = 2.0 - 2.0 * self.zncc
    
    def update_zncc_from_znssd(self):
        """根据ZNSSD更新ZNCC值"""
        self.zncc = 0.5 * (2.0 - self.znssd)


class Result2DS:
    """立体DIC结果，对应C++中的Result2DS"""
    def __init__(self, r1r2_zncc=0.0, r1t1_zncc=0.0, r1t2_zncc=0.0, 
                 r2_x=0.0, r2_y=0.0, t1_x=0.0, t1_y=0.0, t2_x=0.0, t2_y=0.0):
        self.r1r2_zncc = r1r2_zncc
        self.r1t1_zncc = r1t1_zncc
        self.r1t2_zncc = r1t2_zncc
        self.r2_x = r2_x
        self.r2_y = r2_y
        self.t1_x = t1_x
        self.t1_y = t1_y
        self.t2_x = t2_x
        self.t2_y = t2_y
    
    def as_array(self):
        """将结果参数作为数组返回"""
        return np.array([self.r1r2_zncc, self.r1t1_zncc, self.r1t2_zncc, 
                          self.r2_x, self.r2_y, self.t1_x, self.t1_y, self.t2_x, self.t2_y], 
                         dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入结果参数"""
        (self.r1r2_zncc, self.r1t1_zncc, self.r1t2_zncc, 
         self.r2_x, self.r2_y, self.t1_x, self.t1_y, self.t2_x, self.t2_y) = arr


class DeformationVector3D:
    """3D变形模型，对应C++中的DeformationVector3D"""
    def __init__(self, u=0.0, ux=0.0, uy=0.0, uz=0.0,
                 v=0.0, vx=0.0, vy=0.0, vz=0.0,
                 w=0.0, wx=0.0, wy=0.0, wz=0.0):
        self.u = u
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.v = v
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.w = w
        self.wx = wx
        self.wy = wy
        self.wz = wz
    
    def as_array(self):
        """将变形参数作为数组返回"""
        return np.array([self.u, self.ux, self.uy, self.uz,
                          self.v, self.vx, self.vy, self.vz,
                          self.w, self.wx, self.wy, self.wz], dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入变形参数"""
        (self.u, self.ux, self.uy, self.uz,
         self.v, self.vx, self.vy, self.vz,
         self.w, self.wx, self.wy, self.wz) = arr
    
    def warp(self, point):
        """计算变形后的3D坐标"""
        x, y, z = point.x, point.y, point.z
        warped_x = x + self.u + self.ux * x + self.uy * y + self.uz * z
        warped_y = y + self.v + self.vx * x + self.vy * y + self.vz * z
        warped_z = z + self.w + self.wx * x + self.wy * y + self.wz * z
        return Point3D(warped_x, warped_y, warped_z)


class DisplacementVector3D:
    """3D位移向量，对应C++中的DisplacementVector3D"""
    def __init__(self, u=0.0, v=0.0, w=0.0):
        self.u = u
        self.v = v
        self.w = w
    
    def as_array(self):
        """将位移参数作为数组返回"""
        return np.array([self.u, self.v, self.w], dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入位移参数"""
        self.u, self.v, self.w = arr


class StrainVector3D:
    """3D应变模型，对应C++中的StrainVector3D"""
    def __init__(self, exx=0.0, eyy=0.0, ezz=0.0, exy=0.0, eyz=0.0, ezx=0.0):
        self.exx = exx
        self.eyy = eyy
        self.ezz = ezz
        self.exy = exy
        self.eyz = eyz
        self.ezx = ezx
    
    def as_array(self):
        """将应变参数作为数组返回"""
        return np.array([self.exx, self.eyy, self.ezz, self.exy, self.eyz, self.ezx], dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入应变参数"""
        self.exx, self.eyy, self.ezz, self.exy, self.eyz, self.ezx = arr


class Result3D:
    """3D DIC结果，对应C++中的Result3D"""
    def __init__(self, u0=0.0, v0=0.0, w0=0.0, zncc=0.0, iteration=0.0, convergence=0.0, feature=0.0):
        self.u0 = u0
        self.v0 = v0
        self.w0 = w0
        self.zncc = zncc
        self.znssd = 0.0
        self.iteration = iteration
        self.convergence = convergence
        self.feature = feature
    
    def as_array(self):
        """将结果参数作为数组返回"""
        return np.array([self.u0, self.v0, self.w0, self.zncc, self.iteration, self.convergence, self.feature], 
                         dtype=np.float32)
    
    def from_array(self, arr):
        """从数组导入结果参数"""
        self.u0, self.v0, self.w0, self.zncc, self.iteration, self.convergence, self.feature = arr
    
    def update_znssd_from_zncc(self):
        """根据ZNCC更新ZNSSD值"""
        self.znssd = 2.0 - 2.0 * self.zncc
    
    def update_zncc_from_znssd(self):
        """根据ZNSSD更新ZNCC值"""
        self.zncc = 0.5 * (2.0 - self.znssd)


class POI2D:
    """2D兴趣点类，对应C++中的POI2D"""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.deformation = DeformationVector2D()
        self.result = Result2D()
        self.strain = StrainVector2D()
        self.subset_radius = Point2D(0, 0)
    
    def clear(self):
        """重置数据（除位置外）"""
        self.deformation = DeformationVector2D()
        self.result = Result2D()
        self.strain = StrainVector2D()
        self.subset_radius = Point2D(0, 0)
    
    def update_strain_from_deformation(self):
        """根据变形参数计算应变"""
        self.strain.exx = self.deformation.ux
        self.strain.eyy = self.deformation.vy
        self.strain.exy = 0.5 * (self.deformation.uy + self.deformation.vx)


class POI2DS:
    """立体DIC兴趣点类，对应C++中的POI2DS"""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.deformation = DisplacementVector3D()
        self.result = Result2DS()
        self.ref_coor = Point3D()
        self.tar_coor = Point3D()
        self.strain = StrainVector3D()
        self.subset_radius = Point2D(0, 0)
    
    def clear(self):
        """重置数据（除位置外）"""
        self.deformation = DisplacementVector3D()
        self.result = Result2DS()
        self.ref_coor = Point3D()
        self.tar_coor = Point3D()
        self.strain = StrainVector3D()
        self.subset_radius = Point2D(0, 0)


class POI3D:
    """3D兴趣点类，对应C++中的POI3D"""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.deformation = DeformationVector3D()
        self.result = Result3D()
        self.strain = StrainVector3D()
        self.subset_radius = Point3D(0, 0, 0)
    
    def clear(self):
        """重置数据（除位置外）"""
        self.deformation = DeformationVector3D()
        self.result = Result3D()
        self.strain = StrainVector3D()
        self.subset_radius = Point3D(0, 0, 0) 