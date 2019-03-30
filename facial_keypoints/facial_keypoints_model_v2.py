#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as fun


class FacialKpModel(nn.Module):

    def __init__(self):

        super(FacialKpModel, self).__init__()

        # (1, 224, 224)
        # (32, 220, 220)
        # (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 16, 5)

        # (32, 110, 110)
        # (64, 107, 107)
        # (64, 53, 53)
        self.conv2 = nn.Conv2d(16, 32, 4)

        # (64, 53, 53)
        # (128, 51, 51)
        # (128, 25, 25)
        self.conv3 = nn.Conv2d(32, 48, 3)

        # (128, 25, 25)
        # (256, 23, 23)
        # (256, 11, 11)
        self.conv4 = nn.Conv2d(48, 64, 3)

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(7744, 4000)

        self.fc2 = nn.Linear(4000, 1024)

        self.fc3 = nn.Linear(1024, 136)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        x = self.pool(self.conv1(x))
        x = fun.leaky_relu(x)

        x = self.pool(self.conv2(x))
        x = fun.leaky_relu(x)
        x = self.dropout(x)

        x = self.pool(self.conv3(x))
        x = fun.leaky_relu(x)
        x = self.dropout(x)

        x = self.pool(self.conv4(x))
        x = fun.leaky_relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(fun.relu(self.fc1(x)))
        x = self.dropout(fun.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

