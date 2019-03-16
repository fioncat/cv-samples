#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
import numpy as np
from model import Classifier


class TrainConfiguration(object):
    features_path = None

    save_path = None

    epochs = None

    train_features = []

    valid_size = None

    name = None

    show_pre_epoch = 10


def train(conf):
    n_batches = np.load(os.path.join(conf.features_path, 'number.npy'))[0]
    n_features = len(conf.train_features)

    all_train_losses = []
    all_train_acc = []

    all_valid_losses = []
    all_valid_acc = []

    model = Classifier(n_features)

    graph = tf.Graph()
    with graph.as_default():
        model.build(is_training=True)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:

        sess.run(model.init_op)

        for epoch in range(conf.epochs):

            for i in range(n_batches):

                features = None
                labels = None
                for train_feature in conf.train_features:

                    features_path = os.path.join(conf.features_path,
                                                 train_feature,
                                                 'batch_{}'.format(i))

                    feature = np.load(os.path.join(features_path,
                                                   'features.npy'))
                    if labels is None:
                        labels = np.load(os.path.join(features_path,
                                                      'labels.npy'))

                    if features is None:
                        features = feature
                    else:
                        features = np.hstack((features, feature))

                valid_features = features[:conf.valid_size]
                valid_labels = labels[:conf.valid_size]
                train_features = features[conf.valid_size:]
                train_labels = labels[conf.valid_size:]

                train_loss, _, train_acc = \
                    sess.run([model.loss, model.train_op, model.accuracy],
                             feed_dict={model.inputs: train_features,
                                        model.labels: train_labels})
                valid_loss, valid_acc = \
                    sess.run([model.loss, model.accuracy],
                             feed_dict={model.inputs: valid_features,
                                        model.labels: valid_labels})

                all_train_losses.append(train_loss)
                all_train_acc.append(train_acc)

                all_valid_losses.append(valid_loss)
                all_valid_acc.append(valid_acc)

            if (epoch + 1) % conf.show_pre_epoch == 0:
                print('Epoch {}/{}, loss={:.4f} acc={:.2f}'
                      ' val_loss={:.4f} val_acc={:.2f}'.format(epoch, conf.epochs,
                                                               all_train_losses[-1],
                                                               all_train_acc[-1],
                                                               all_valid_losses[-1],
                                                               all_valid_acc[-1]))
        complete_save_path = os.path.join(conf.save_path, conf.name)
        if not os.path.isdir(complete_save_path):
            os.mkdir(complete_save_path)
        complete_save_path = os.path.join(complete_save_path, 'catdog.ckpt')
        saver.save(sess, complete_save_path)

        print('Saved at', complete_save_path)

    return all_train_losses, all_train_acc, all_valid_losses, all_valid_acc
