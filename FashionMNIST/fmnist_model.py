#!/usr/bin/python
# -*- coding=UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as fun


class CNN(nn.Module):
    """
    用于FashionMNIST分类的CNN模型.

    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 3)

        self.fc1 = nn.Linear(20 * 5 * 5, 50)

        self.fc_drop = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        # x: (1, 28, 28) -> (10, 26, 26) -> (10, 13, 13)
        x = self.pool(fun.leaky_relu(self.conv1(x)))

        # x: (10, 13, 13) -> (20, 11, 11) -> (20, 5, 5)
        x = self.pool(fun.leaky_relu(self.conv2(x)))

        # Flat, x -> (20 * 5 * 5)
        x = x.view(x.size(0), -1)

        # FC层
        x = fun.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)

        x = fun.log_softmax(x, dim=1)

        return x


if __name__ == '__main__':

    net = CNN()

    print(net)
