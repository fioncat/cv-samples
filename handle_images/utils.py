#!/usr/bin/python
# -*- coding: utf-8 -*-

# 对 OpenCV 进行一些简单的二次封装
# 提供一些简单的图像处理函数

import numpy as np
import cv2

import matplotlib.pyplot as plt


def imread(path, mode='RGB'):
    """
        读取图像, 以多维向量返回

    Args:
        path: 图像路径
        mode: 'RGB': 以RGB模式读取

    """
    image = cv2.imread(path)
    if mode == 'RGB':
        cvt = cv2.COLOR_BGR2RGB
    return cv2.cvtColor(image, cvt)


def rgb2gray(image):
    """
        将RGB图像转换为灰度图像
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normal_image(image):
    """
        将图像像素值归一到0-1
    """
    return image / 255.0


def ft_image(normaled_image):
    """
        对图像做FT. 以可视化图像的形式返回.
        注意传入的图像必须是已经归一化的灰度图像.
    """
    f = np.fft.fft2(normaled_image)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift))


def show_images_in_single_row(images, titles, cmaps=None, figsize=(10, 5)):
    """
        在单行显示多张图像.
    """

    _, axs = plt.subplots(1, len(images), figsize=figsize)

    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        if cmaps is not None:
            cmap = cmaps[i]
        else:
            cmap = None
        ax.imshow(images[i], cmap=cmap)
