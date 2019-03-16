#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import csv

from model import Classifier


def test(features_path, save_path, test_features, csv_path, name):
    model = Classifier(len(test_features))
    n_batches = np.load(os.path.join(features_path, 'number.npy'))[0]

    graph = tf.Graph()

    with graph.as_default():
        model.build(is_training=False)
        saver = tf.train.Saver()

    model_path = os.path.join(save_path, name, 'catdog.ckpt')

    results = {}
    count = 0
    with tf.Session(graph=graph) as sess:

        saver.restore(sess, model_path)

        for i in range(n_batches):
            features = None
            labels = None
            for test_feature in test_features:

                feature_path = os.path.join(features_path,
                                            test_feature,
                                            'batch_{}'.format(i))

                feature = np.load(os.path.join(feature_path,
                                               'features.npy'))
                if labels is None:
                    labels = np.load(os.path.join(feature_path,
                                                  'labels.npy'))

                if features is None:
                    features = feature
                else:
                    features = np.hstack((features, feature))

            prediction = sess.run(model.prediction, feed_dict={
                model.inputs: features})

            for j, label in enumerate(labels):
                results[label] = prediction[j]
                count += 1
    print('Processed {} images'.format(count))
    keys = sorted(results.keys())

    with open(os.path.join(csv_path, name + '.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for key in keys:
            writer.writerow([key, results[key]])
