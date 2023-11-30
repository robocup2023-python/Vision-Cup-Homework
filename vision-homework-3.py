import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
# 1 读取图像
img = cv.imread("./samoyed.jpeg",0)
# 2 傅里叶变换
# 2.1 正变换
dft = cv.dft(np.float64(img),flags = cv.DFT_COMPLEX_OUTPUT) 
# 2.2 频谱中心化
dft_shift = np.fft.fftshift(dft)
# 2.3 计算频谱和相位谱
mag, angle = cv.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1], angleInDegrees=True)
mag=20*np.log(mag) 
 
rows, cols = img.shape 
crow, ccol = int(rows/2) , int(cols/2) 
mask = np.zeros((rows, cols,2), np.uint8) 
#两个通道，与频域图像匹配 
mask[crow-30:crow+30, ccol-30:ccol+30,:] = 1 
fShift = dft_shift*mask 
ishift = np.fft.ifftshift(fShift) 
iImg = cv.idft(ishift) 
iImg= cv.magnitude(iImg[:, :,0], iImg[:, :,1])
 
# 4 图像显示
plt.figure(figsize=(10,8))
plt.subplot(221),plt.imshow(img, cmap = 'gray') 
plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(222),plt.imshow(mag, cmap = 'gray')
plt.title('Mag'), plt.xticks([]), plt.yticks([]) 
plt.subplot(223),plt.imshow(angle, cmap = 'gray')
plt.title('Angle'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(iImg, cmap = 'gray')
plt.title('Back'), plt.xticks([]), plt.yticks([])
plt.show()