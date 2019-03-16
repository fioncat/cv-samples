#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


class Classifier(object):

    def __init__(self, n_features):
        self.n_features = n_features

        self.inputs = None
        self.labels = None
        self.prediction = None
        self.loss = None

        self.train_op = None
        self.accuracy = None

        self.init_op = None

    def build(self, is_training=True):

        input_n = self.n_features * 2048 if self.n_features < 4 else 6656

        self.inputs = tf.placeholder(tf.float32, [None, input_n])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        if is_training:
            x = tf.nn.dropout(self.inputs, 0.5)
        else:
            x = self.inputs

        logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        logits = tf.reshape(logits, [-1])
        self.prediction = tf.sigmoid(logits)

        if is_training:
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=self.labels)
            self.loss = tf.reduce_mean(self.loss)

            self.train_op = tf.train.RMSPropOptimizer(learning_rate=2e-5).minimize(loss=self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
                self.prediction > 0.5, self.labels > 0), tf.float32))

            self.init_op = tf.global_variables_initializer()
