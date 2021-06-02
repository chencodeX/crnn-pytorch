#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2020 - zihao.chen
'''
@Author : zihao.chen
@File : data_augument.py
@Create Date : 2020/11/2
@Descirption :
'''
import os
import cv2
import numpy as np
import random
from skimage.filters import threshold_local


def get_emojis():
    emoji_path = "./selene/datasets/emojis/"
    all_files = os.listdir(emoji_path)
    all_paths = []
    for file_name in all_files:
        if not file_name.endswith('.png'):
            continue
        temp_img = cv2.imread(emoji_path + file_name, cv2.IMREAD_UNCHANGED)
        MASK = temp_img[..., 3]
        COLOR = temp_img[..., :3]
        COLOR[MASK == 0, :] = 0
        temp_img = cv2.cvtColor(COLOR, cv2.COLOR_RGB2GRAY)
        temp_img[MASK < 50] = 255
        all_paths.append(temp_img)

    return all_paths


# 锐化
def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


# 椒盐噪声
def pepper_and_salt(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 高斯噪声
def gaussian_noise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 高斯模糊
def gaussian_blur(src, sigma=1.5):
    if src.shape[0] < 12:
        return src
    elif src.shape[0] <= 24:
        kernel_size = (3, 3)
    else:
        kernel_size = (5, 5)
    img = cv2.GaussianBlur(src, kernel_size, sigma)
    return img


# 随机crop以应对检测网络奇奇怪怪的高度误差
def random_crop(src):
    h, w = src.shape
    if h <= 16:
        return src
    if h <= 24:
        scale = random.randint(-1, 1)
    else:
        scale = random.randint(-2, 2)
    scale *= max((h // 32), 1)
    if scale > 0:
        x = random.randint(-scale, scale)
        img2 = cv2.copyMakeBorder(src, scale - x, scale + x, 0, 0, cv2.BORDER_REPLICATE)
        return img2
    elif scale == 0:
        x = random.randint(-2, 2)
        if x >= 0:
            src = src[x:]
            img2 = cv2.copyMakeBorder(src, 0, x, 0, 0, cv2.BORDER_REPLICATE)
            return img2
        else:
            src = src[:x]
            img2 = cv2.copyMakeBorder(src, abs(x), 0, 0, 0, cv2.BORDER_REPLICATE)
            return img2
    else:
        x = random.randint(0, abs(scale) * 2)
        src = src[x:h + 2 * (scale) + x, :]
        return src[x:h + 2 * (scale) + x, :]


# 拉普拉斯锐化
def laplacian_sharpen(src, ks=3):
    gray_lap = cv2.Laplacian(src, cv2.CV_16S, ksize=ks)
    dst = cv2.convertScaleAbs(gray_lap)
    return dst


# 模拟扫描的黑白效果
def scan(src):
    block_size = 11
    offset = random.randint(6, 9)
    T = threshold_local(src, block_size, offset=offset, method="gaussian")
    src = (src > T).astype("uint8") * 255
    return src


# 对比度增强
def normalize(src):
    img_norm = cv2.normalize(src, dst=None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    return img_norm


# 随机腐蚀部分区域
def random_erode(src):
    if src.shape[0] < 24:
        return src
    elif src.shape[0] <= 48:
        kernel_size = (2, 2)
    else:
        kernel_size = (3, 3)

    select_max_num = src.shape[1] // 32 // 3
    if select_max_num < 1:
        return src
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.dilate(src, kernel, iterations=1)
    select_num = np.random.randint(select_max_num // 2, select_max_num + 1)
    for i in range(select_num):
        index = np.random.randint(0, src.shape[1])
        start = max(0, index - 16)
        end = min(src.shape[1], index + 16)
        src[:, start:end] = erosion[:, start:end]
    return src


# 主要针对反色误识别问题
def random_invert(src):
    if random.random() < 0.3:
        max_value = src.max()
    else:
        max_value = 255
    if src.shape[1] <= 48:
        return max_value - src
    if random.random() < 0.3:
        index = np.random.randint(0, src.shape[1] - 16)
        index = max(index, 16)
        if random.random() < 0.5:
            src[:, :index] = max_value - src[:, :index]
        else:
            src[:, index:] = max_value - src[:, index:]
    else:
        src = max_value - src
    return src


if __name__ == '__main__':
    import os

    all_imgs = os.listdir("/Users/moka/Downloads/download_scp/reg_samples")
