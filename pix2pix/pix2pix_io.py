#!/usr/bin/python

"""
facades数据集的IO操作.
本代码主要目的是创建facades的tf.data.Dataset对象.

"""

import tensorflow as tf
import numpy as np
import os

# 图像尺寸
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3

# 数据集对象参数
BUFFER_SIZE = 100
BATCH_SIZE = 1


def load_image(image_file, is_training):
    """
    原始图像由真实图像和输入图像拼接而成. 需要对原始图像分割.
    如果在训练, 对图像进行随机的增强操作.

    Args:
        image_file: 图像路径
        is_training: 是否在训练
    """

    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    # 切割原始图像, 提取出输入图像和真实图像
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_training:

        input_image = tf.image.resize_images(input_image, [286, 286],
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, [286, 286],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 随机地切割图像        
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        # 有50%的几率翻转图像
        if np.random.random() > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

    else:

        input_image = tf.image.resize_images(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                             align_corners=True, method=2)
        real_image = tf.image.resize_images(real_image, [IMG_HEIGHT, IMG_WIDTH],
                                            align_corners=True, method=2)
    
    # 将像素值归一到 [-1,1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def download_and_get_dataset(base_path='data/'):
    """
    如果数据集不存在, 下载并提取数据集, 随后生成对应的tf.data.Dataset对象.
    如果数据集已经在磁盘上了, 直接生成tf.data.Dataset对象

    Args:
        base_path: 存放数据集的基础目录. 其中应该包括:
                        1. facades.tar.gz: 数据集的压缩包
                        2. facades/: 存放训练, 测试, 验证数据
                   若以上目录不存在, 此调用会自动下载
    Returns:
        (train_datset, test_dataset)
        分别表示训练和测试数据集对象
    """
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                          cache_subdir=os.path.abspath(base_path),
                                          origin='https://people.eecs.berkeley.edu/'
                                                 '~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
                                          extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    train_dataset = tf.data.Dataset.list_files(path + 'train/*.jpg')
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.map(lambda x: load_image(x, True))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(path + 'test/*.jpg')
    test_dataset = test_dataset.shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.map(lambda x: load_image(x, False))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_data, test_data = download_and_get_dataset()
