#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as fun


class FacialKpModel(nn.Module):

    def __init__(self):

        super(FacialKpModel, self).__init__()

        # Input: (1, 96, 96)
        # (96 - 4) / 1 + 1 = 93
        # Conv Output: (32, 93, 93)
        # Pool Output: (32, 46, 46)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv1_dropout = nn.Dropout(p=0.3)

        # Input: (32, 46, 46)
        # (46 - 3) / 1 + 1 = 44
        # Conv Output: (64, 44, 44)
        # Pool Output: (64, 22, 22)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_dropout = nn.Dropout(p=0.3)

        # Input: (64, 22, 22)
        # (22 - 2) / 1 + 1 = 21
        # Conv Output: (128, 21, 21)
        # Pool Output: (128, 10, 10)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv3_dropout = nn.Dropout(p=0.3)

        # Input: (128, 10, 10)
        # (10 - 1) / 1 + 1 = 10
        # Conv Output: (256, 10, 10)
        # Pool Output: (256, 5, 5)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.conv4_dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(256 * 5 * 5, 1000)
        self.fc1_dropout = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(1000, 200)
        self.fc2_dropout = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(200, 68 * 2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Conv1
        x = self.conv1(x)
        x = fun.leaky_relu(x)
        x = self.pool(x)
        x = self.conv1_dropout(x)

        # Conv2
        x = self.conv2(x)
        x = fun.leaky_relu(x)
        x = self.pool(x)
        x = self.conv2_dropout(x)

        # Conv3
        x = self.conv3(x)
        x = fun.leaky_relu(x)
        x = self.pool(x)
        x = self.conv3_dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = fun.relu(x)
        x = self.fc1_dropout(x)

        # FC1
        x = self.fc2(x)
        x = fun.relu(x)
        x = self.fc2_dropout(x)

        # FC3
        x = self.fc3(x)

        return x

