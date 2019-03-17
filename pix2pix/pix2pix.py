#!/usr/bin/python
# -*- coding=UTF-8 -*-

"""
模型训练代码
官方使用的是eager, 我这里还是改用session进行训练
GAN模型的定义已经实现在pix2pix_model.py中
facades数据集是一个公开的街景数据集, 其下载和读取代码已经实现在pix2pix_io.py
"""

import tensorflow as tf
import time
import matplotlib.pyplot as plt

from pix2pix_model import Generator, Discriminator
from IPython.display import clear_output


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    计算识别器误差.

    Args:
        disc_real_output: 识别器处理真实图像结果
        disc_generated_output: 识别器处理生成图像结果

    Returns:
        识别器误差
    """

    real_loss = \
        tf.losses.sigmoid_cross_entropy(multi_class_labels=
                                        tf.ones_like(disc_generated_output),
                                        logits=disc_real_output)
    generated_loss = \
        tf.losses.sigmoid_cross_entropy(multi_class_labels=
                                        tf.zeros_like(disc_generated_output),
                                        logits=disc_generated_output)
    return real_loss + generated_loss


def generator_loss(disc_generated_output, gen_output, target, _lambda=100):
    """
    计算生成器误差.
    同时会加上L1正则

    Args:
        disc_generated_output:
        gen_output: 识别器产生的图像
        target: 真实图像
        _lambda: L1正则参数

    Returns:
        生成器误差
    """

    gen_loss = \
        tf.losses.sigmoid_cross_entropy(multi_class_labels=
                                        tf.ones_like(disc_generated_output),
                                        logits=disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    return gen_loss + (_lambda * l1_loss)


def _show_images(inp, tar, pred):
    """
    展示图像, 辅助函数

    Args:
        inp: 输入图像
        tar: 目标图像
        pred: 识别器产生的图像
    """

    display_list = [inp, tar, pred]
    titles = ['Input Images', 'Ground Truth', 'Predicted Imgae']

    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.show()


def session_train(train_dataset,
                  epochs,
                  learning_rate=2e-4,
                  checkpoint_dir='./saved_model/pix2pix',
                  test_dataset=None):
    """
    执行训练过程.

    Args:
        train_dataset: 训练数据集, 由pix2pix_io获取
        epochs: 迭代轮数
        learning_rate: 学习率
        checkpoint_dir: 模型保存路径
        test_dataset: 如果不为None, 表示测试数据集, 在训练过程中会实时
                      展示生成器在这个数据集上的表现
    """

    train_dataset = train_dataset.repeat(epochs)

    graph = tf.Graph()
    with graph.as_default():

        # 定义生成器和识别器
        generator = Generator()
        discriminator = Discriminator()

        # 图像迭代器
        ite = train_dataset.make_one_shot_iterator()
        input_image, target = ite.get_next()

        # 生成器产生的假图像
        gen_output = generator(input_image, True)

        # 识别器对假图像和真图像处理产生结果
        disc_real_output = discriminator(input_image, target, training=True)
        disc_generated_output = discriminator(input_image, gen_output, training=True)

        # 计算误差
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # 优化过程
        gen_train_op = \
            tf.train.AdamOptimizer(learning_rate,
                                   beta1=0.5).minimize(gen_loss, var_list=generator.variables)
        disc_train_op = \
            tf.train.AdamOptimizer(learning_rate,
                                   beta1=0.5).minimize(disc_loss, var_list=discriminator.variables)
        saver = tf.train.Saver(var_list=generator.variables)

        # 如果有测试集, 在测试集上作用训练后的生成器
        if test_dataset:
            test_dataset = test_dataset.repeat()
            test_input, test_target = test_dataset.make_one_shot_iterator().get_next()
            prediction = generator(test_input, training=True)
        else:
            test_input = None
            test_target = None
            prediction = None

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())

        try:
            step = 0
            while True:
                start = time.time()

                sess.run([gen_train_op, disc_train_op])

                # 如果有测试集, 展示生成器的效果
                if test_dataset:
                    if step % 1 == 0:
                        clear_output(wait=True)
                        test_input_image, test_target_image, test_pred_image = \
                            sess.run([test_input, test_target, prediction])
                        _show_images(test_input_image[0], test_target_image[0], test_pred_image[0])

                # 每20轮保存一次模型
                if (step + 1) % 20 == 0:
                    saver.save(sess, checkpoint_dir)

                print('step {}, Time taken: {} sec.'.format(step,
                                                            time.time() - start))
                step += 1

        except tf.errors.OutOfRangeError:  # 所有数据迭代完毕
            saver.save(sess, checkpoint_dir)
