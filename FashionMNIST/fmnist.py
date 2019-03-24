#!/usr/bin/python
# -*- coding=UTF-8 -*-

import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn

import numpy as np

from fmnist_model import CNN


def train(epochs, train_loader, learning_rate=1e-3):
    """
    使用FashionMNIST训练CNN模型, 并返回.
    注意这个训练函数不会保存模型.

    如果CUDA可用, 将使用GPU加速.

    Args:
        train_loader: 训练数据加载器
        epochs: 训练轮数
        learning_rate: 学习率

    Returns:
        model, losses
        model: 已经训练好的模型
        losses: 训练过程中记录的损失
    """
    # 定义模型以及损失标准
    model = CNN()
    criterion = nn.NLLLoss()

    # 如果CUDA可用, 启用
    if cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # 定义损失器
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):

        # 统计的总损失
        running_loss = 0.0

        for batch_i, (images, labels) in enumerate(train_loader):

            # 如果CUDA可用, 将数据存到显存中
            if cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # 清空上一次计算的梯度
            optimizer.zero_grad()

            # 图像前向传递
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 通过损失计算梯度
            loss.backward()

            # 优化参数
            optimizer.step()

            # 统计损失
            running_loss += loss.item()

            # 每1000batch, 记录一次损失并打印信息
            if batch_i % 1000 == 999:
                avg_loss = running_loss / 1000
                losses.append(avg_loss)
                print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch + 1,
                                                                  batch_i + 1,
                                                                  avg_loss))
                running_loss = 0.0

    print('Training over')
    return model, losses


def show_losses(losses):
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel('1000\'s of batches')
    plt.ylabel('loss')
    plt.ylim(0, 2.5)

    plt.show()


def test(model, test_loader, classes, batch_size=20):

    test_loss = np.zeros(1)
    class_correct = [0. for _ in range(10)]
    class_total = [0. for _ in range(10)]

    # 将模型设为评估模式, 此时dropout等layer会失效
    model.eval()

    criterion = nn.NLLLoss()

    for batch_i, (images, labels) in enumerate(test_loader):

        if cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # 获得网络的输出
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)
        test_loss += ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))

        _, predicted = np.max(outputs.data, 1)

        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        for i in range(batch_size):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print('Test Loss: {:.5f}'.format(test_loss.numpy()[0]))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                class_correct[i], class_total[i]))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100 * sum(class_correct) / sum(class_total),
        sum(class_correct), sum(class_total) ))


def save_model(model, path='saved_model/fashion_net.pt'):

    torch.save(model.state_dict(), path)
