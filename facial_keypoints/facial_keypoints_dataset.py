#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.image as mimg

from torch.utils.data import Dataset, DataLoader

DATA_URL = r'http://47.101.211.243:8000/dataset/' \
           r'facial-keypoints-data.zip'


def _unzip(zip_path, target_path):
    """
        解压zip文件

    Args:
        zip_path: zip压缩包路径
        target_path: 目标路径

    """
    import zipfile

    with zipfile.ZipFile(zip_path) as file:
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        for name in file.namelist():
            file.extract(name, target_path)


def download_and_unzip_data(
        data_url=DATA_URL,
        zip_file_name='data.zip',
        extract_path='faces',
        base_path='./data'):
    """
        下载并解压面部关键点数据集.
        函数将会下载数据集的压缩包(如果数据集不存在的话)
        如果数据集已经存在于指定路径,则函数不会做任何事情.

    Args:
        data_url: 请求数据集的url
        zip_file_name: 下载的压缩包的储存名称
        extract_path: 解压目录
        base_path: 存放数据的基础目录

    """

    data_path = os.path.join(base_path, extract_path)

    # 如果数据集已经存在了, 不需要请求下载文件, 直接结束函数
    if os.path.isdir(data_path):
        print('数据集已经存在, 跳过下载解压')
        return

    from contextlib import closing
    import requests

    # 如果base目录不存在, 创建
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    # 下载数据集并解压
    with closing(requests.get(data_url, stream=True)) as response:

        print('请求:', DATA_URL)

        chunk_size = 1024
        content_size = int(response.headers['content-length'])

        print('数据大小: {} bytes.'.format(content_size))

        download = True
        target_path = os.path.join(base_path, zip_file_name)

        # 如果压缩包已经存在并且大小和远程文件相等, 则不必下载
        # 压缩比不存在或者大小不相符, 删除存在的文件并下载
        if os.path.exists(target_path):
            origin_file_size = os.path.getsize(target_path)
            if origin_file_size == content_size:
                print('数据已经存在, 不再下载')
                download = False
            else:
                print('数据不完整, 重新下载')
                os.remove(target_path)
        else:
            print('数据不存在, 开始下载')

        if download:

            with open(target_path, 'wb') as file:

                # 从网络下载文件
                received = 0
                for data in response.iter_content(
                        chunk_size=chunk_size):

                    file.write(data)
                    received += len(data)

                    info = '已接收: {:>10.2f} MiB, 剩余: {:>10.2f} MiB\r'.format(
                        received / 1024 / 1024,
                        (content_size - received) / 1024 / 1024)

                    sys.stdout.write(info)
                    sys.stdout.flush()

        # 解压压缩包
        print('解压数据中...')
        _unzip(target_path, data_path)
        print('解压完毕!')


class FacialKeypointsDataset(Dataset):
    """
        脸部关键点数据集基础类.

        数据集包括数据目录和一个csv文件. 其中数据目录存放图片, csv存放图像
        名称和关键点坐标的映射.

        训练数据和测试数据是分开的, 它们的类继承自此类, 因为它们有自己的
        图像目录和csv文件.
    """

    def __init__(self, base_dir, csv_name, data_name, transform=None):
        """

        Args:
            base_dir: 存放数据集的基础目录
            csv_name: 记录关键点和图像映射的csv文件路径
            data_name: 存放图像的文件路径
            transform: 如果存在, 在返回数据的时候会对sample调用此函数.
                       不存在则不会对sample进行操作.
        """

        download_and_unzip_data(base_path=base_dir)
        base_dir = os.path.join(base_dir, 'faces')

        self.data_path = os.path.join(base_dir, data_name)

        csv_path = os.path.join(base_dir, csv_name)
        self.kp_frames = pd.read_csv(csv_path)

        self.transform = transform

    def __len__(self):
        return len(self.kp_frames)

    def __getitem__(self, item):
        """
        返回的sample格式(如果没有经过transform处理)

        {
            'image': image,
            'kps': kps
        }

        image表示当期图像, 已经读取为shape为 [?, ?, 3] 的numpy数组
        kps表示图像的关键点, 将以矩阵的形式储存

        """

        image_name = os.path.join(self.data_path,
                                  self.kp_frames.iloc[item, 0])

        image = mimg.imread(image_name)

        # 如果图像有alpha颜色通道, 舍弃之
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        kps = self.kp_frames.iloc[item, 1:].values
        kps = kps.astype('float').reshape(-1, 2)

        sample = {'image': image, 'kps': kps}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TrainDataset(FacialKeypointsDataset):
    """
        训练数据集类.
        直接迭代这个对象可以产生训练数据.
    """

    def __init__(self, base_dir='./data', transform=None):
        FacialKeypointsDataset.__init__(self, base_dir,
                                        'training_frames_keypoints.csv',
                                        'training',
                                        transform=transform)


class TestDataset(FacialKeypointsDataset):
    """
        测试数据集类.
    """

    def __init__(self, base_dir='./data', transform=None):
        FacialKeypointsDataset.__init__(self, base_dir,
                                        'test_frames_keypoints.csv',
                                        'test',
                                        transform=transform)


def get_loader(dataset, batch_size):
    """
        根据dataset生成data loader. 会进行batch和随机的shuffle操作.

    Args:
        dataset: 原始数据集对象
        batch_size:

    Returns:
        data loader
    """

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def normalize(sample):
    """
        图像归一化.
        先将图像转换为灰度图(彩色信息对于人脸模型来说没有用)
        然后将像素值范围归一到[0,1]之间.

    Args:
        sample: 要归一化的样本

    Returns:
        归一化之后的样本

    """

    image, kps = sample['image'], sample['kps']

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image / 255.0
    kps = (kps - 100) / 50.0

    return {'image': image, 'kps': kps}


def rescale(sample, size):
    """
        将图像重新缩放到目标尺寸.

    Args:
        sample: 要缩放的样本
        size: 目标尺寸. 如果是tuple或list, 则会直接把图像缩放到这个尺寸.
              如果是int, 较小的图像边缘会匹配到size以保持纵横比相等.

    Returns:
        缩放后的sample

    """

    image, kps = sample['image'], sample['kps']

    h, w = image.shape[:2]
    if isinstance(size, int):
        # 重构较小的边缘, 使纵横比和原图相等
        if h > w:
            nh, nw = size * h / w, size
        else:
            nh, nw = size, size * w / h

    elif isinstance(size, (tuple, list)) and len(size) == 2:
        # 直接缩放成用户想要的尺寸
        nh, nw = size[0], size[1]
    else:
        raise_size_error(size)
        return

    nh, nw = int(nh), int(nw)

    # 图像和关键点都需要缩放
    image = cv2.resize(image, (nw, nh))
    kps = kps * [nw / w, nh / h]

    return {'image': image, 'kps': kps}


def random_crop(sample, size):
    """
        随机裁剪样本中的图像

    Args:
        sample: 要裁剪的样本
        size: 裁剪后的大小. 如果是int, 将会进行矩形裁剪.

    Returns:
        裁剪后的样本.

    """

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        pass
    else:
        raise_size_error(size)

    image, kps = sample['image'], sample['kps']

    h, w = image.shape[:2]
    nh, nw = size[0], size[1]

    top = np.random.randint(0, h - nh)
    left = np.random.randint(0, w - nw)

    image = image[top:top + nh, left:left + nw]

    kps = kps - [left, top]

    return {'image': image, 'kps': kps}


def to_tensor(sample):
    """
        将sample中的ndarray转换为torch可识别的tensor

    Args:
        sample: 要转换的sample

    Returns:
        转换后的sample

    """

    image, kps = sample['image'], sample['kps']

    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)

    image = image.transpose((2, 0, 1))

    return {'image': torch.from_numpy(image),
            'kps': torch.from_numpy(kps)}


def raise_size_error(size):
    """当size不是int, tuple或list, 或size的长度不为2, 抛出异常"""

    raise ValueError('size must be an int, tuple or '
                     'list(with len equals to 2).'
                     'Found type {}, len = {}'.
                     format(type(size), len(size)))


def show_kps(image, kps):
    """
        在图像中显示关键点.

    Args:
        image: 要显示的图像
        kps: 关键点坐标
    """
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.scatter(kps[:, 0], kps[:, 1], s=20, marker='.', c='m')


def train_transform(sample):
    """
        对训练数据的预处理转换. 这个函数会进行下面这些调用:
            1. rescale(100): 按照100对图像进行缩放
            2. random_crop(96): 按照96对图像裁剪,
                                 裁剪后的图像尺寸为(96, 96)
            3. normalize(): 对图像进行灰度化和归一化,
                            处理后像素值在[0, 1]之间, 并且是灰度值

    Args:
        sample: 要处理的样本

    Returns:
        转换后的样本

    """
    sample = rescale(sample, 100)
    sample = random_crop(sample, 96)
    sample = normalize(sample)
    return to_tensor(sample)
    # return sample
