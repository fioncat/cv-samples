#!/usr/bin/python
# -*- coding=UTF-8 -*-

from torch.utils.data import DataLoader

import torchvision as tv
import numpy as np


def load_fmnist_data(data_path='./data', batch_size=20):
    """
    加载FMNIST数据.
    如果在磁盘上没有找到数据, 则会自动下载

    Args:
        data_path: 数据元目录
        batch_size:

    Returns:
        train_loader, test_loader, classes
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        classes: 分类对照表

    """

    data_transform = tv.transforms.ToTensor()

    train_data = tv.datasets.FashionMNIST(root=data_path,
                                          train=True,
                                          download=True,
                                          transform=data_transform)

    test_data = tv.datasets.FashionMNIST(root=data_path,
                                         train=False,
                                         download=True,
                                         transform=data_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_loader, test_loader, classes


def show_samples(batch_size=20):
    """
    显示一个batch的FashionMNIST图像

    Args:
        batch_size:

    """

    import matplotlib.pyplot as plt

    train_loader, _, classes = load_fmnist_data(batch_size=batch_size)

    data_iter = iter(train_loader)

    images, labels = data_iter.next()
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in range(batch_size):
        ax = fig.add_subplot(2, batch_size / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(classes[labels[idx]])

    plt.show()


if __name__ == '__main__':

    show_samples()
