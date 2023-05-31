import cv2
import numpy as np
import math

"""
应用中,img1是Imgae类型的数据,img2是numpy类型的数据;
需要对其进行数值转换：
    将数据范围从[0, 1]调整到[0, 255]。
    将数据类型转换为uint8。
    将NumPy数组转换为PIL Image对象。
"""

def calculate_l1(img1, img2):

    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    l1 = np.mean(np.abs(img1 - img2))

    return l1

def calculate_psnr(img1, img2):

    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')

    # 从Image类型转换为numpy数组
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

