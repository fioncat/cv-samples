#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn


def train(model,
          train_loader,
          epochs,
          batch_size=10,
          learning_rate=0.001):

    criterion = nn.SmoothL1Loss()
    model.train()

    if cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate)

    for epoch in range(epochs):

        count = 0
        total_loss = 0
        for batch_i, sample in enumerate(train_loader):

            images = sample['image']
            kps = sample['kps']
            kps = kps.view(kps.size(0), -1)

            kps = kps.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            if cuda.is_available():
                images = images.cuda()
                kps = kps.cuda()

            output_kps = model(images)

            loss = criterion(output_kps, kps)
            total_loss += loss.item()
            count += 1

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch {:>2d}/{}'
                  ' Loss={:<2.6f}'.format(epoch + 1, epochs,
                                          total_loss /
                                          (count * batch_size)))

    print('Training over.')


def save_model(model,
               path='saved_model/facial_keypoints_net.pt'):

    torch.save(model.state_dict(), path)


def load_model(path='saved_model/facial_keypoints_net.pt'):

    from facial_keypoints_model_v2 import FacialKpModel

    model = FacialKpModel()
    model.load_state_dict(torch.load(path))

    return model
