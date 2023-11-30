# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2

class FirstHomework:
    def __init__(self):
        self.sigma = 1.3
        # 核的边长必须是奇数
        self.kernel_size = 21 
        self.img_path = "./samoyed.jpeg"
        self.K = np.zeros((self.kernel_size,self.kernel_size),dtype=np.float)

    # 1.手写高斯滤波
    def GaussianFilter(self):
        img = cv2.imread(self.img_path)
        h,w,c = img.shape
        
        # 零填充
        pad = self.kernel_size//2
        out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
        out[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
        
        # 定义滤波核
        self.K = np.zeros((self.kernel_size,self.kernel_size),dtype=np.float)
        
        for x in range(-pad,-pad+self.kernel_size):
            for y in range(-pad,-pad+self.kernel_size):
                self.K[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(self.sigma**2)))
        self.K /= (self.sigma*np.sqrt(2*np.pi))
        
        # 归一化
        self.K /=  self.K.sum()
        # print(self.K)
        # 卷积的过程
        tmp = out.copy()
        for y in range(h):
            for x in range(w):
                for ci in range(c):
                    out[pad+y,pad+x,ci] = np.sum(self.K*tmp[y:y+self.kernel_size,x:x+self.kernel_size,ci])
        
        out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
        
        return out

    # 2.cv库调用高斯滤波
    def Cv2_Gauss(self):
        img = cv2.imread(self.img_path)
        # source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
        return result

    # 3.画高斯滤波核
    def Draw_Gauss_Kernel(self):
        x,y = np.mgrid[0:self.kernel_size:1,0:self.kernel_size:1]
        z = self.K
        #绘制曲面图
        plt.figure()
        ax = plt.axes(projection='3d')
        #调用plot_surface()函数
        ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
        ax.set_title('Surface plot')
        plt.show()
    
    # 4.padding zeros填充
    def boundaryZeroPadding(self):
        img = cv2.imread(self.img_path)
        # 获取图片的高，宽
        img_height, img_width, img_channel = img.shape
        # 获得 K 值
        kx, ky = self.kernel_size//2, self.kernel_size//2
      
        # 左右需要填充的 zero
        # 此处必须加上 dtype=np.uint8，否则 cv2 float 显示全白
        zeros_array = np.zeros((img_height, kx,img_channel), dtype=np.uint8)
 
        # 给图片左右两边添加 0
        img_copy = np.concatenate([zeros_array, img, zeros_array], axis=1)

        # 上下需要填充的 zero
        zeros_array = np.zeros((ky, 2*kx + img_width,img_channel), dtype=np.uint8)
        # 给图片上下两边添加 0
        img_result = np.concatenate([zeros_array, img_copy, zeros_array], axis=0)
        return img_result

    # 5.padding WrapAround填充
    def boundaryWrapAround(self):
        img = cv2.imread(self.img_path)
        # 获取图片的高，宽
        # img_height, img_width,img_channel = img.shape
        # 获得 K 值
        kx, ky = self.kernel_size//2, self.kernel_size//2
        '''
        块复制 位置分布定义, X为原图片
        E |     D     | F
        --------------------
            | 1 | 2 | 3 |
            | - | - | - |
        A | 4 | X | 5 | B
            | - | - | - |
            | 6 | 7 | 8 |
        --------------------
        G |     C     | H
        '''
        # 求上下
        part_D = img[-ky:]
        part_C = img[: ky]
        # 中间合并
        part_DXC = np.concatenate([part_D, img, part_C], axis=0)
        # 复制 DXC 的 右边
        part_EAG = part_DXC[:, -kx:]
        # 复制 DXC 的 左边
        part_FBH = part_DXC[:,  :kx]
        # 整体合并
        img_result = np.concatenate([part_EAG, part_DXC, part_FBH], axis=1)
        return img_result


if __name__ == "__main__":
    work = FirstHomework()
    # 高斯滤波-手写
    cv2.imshow('Gaussian-Write',work.GaussianFilter())
    cv2.waitKey(0)

    # 高斯滤波-cv库
    cv2.imshow('Gaussian-CV',work.Cv2_Gauss())
    cv2.waitKey(0)

    #padding- zeros填充
    cv2.imshow('Padding-Zeros',work.boundaryZeroPadding())
    cv2.waitKey(0)

    #padding- wrap around填充
    cv2.imshow('Padding-WrapAround',work.boundaryWrapAround())
    cv2.waitKey(0)

    # 绘制高斯滤波核
    work.Draw_Gauss_Kernel()
    
    
