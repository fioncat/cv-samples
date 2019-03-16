#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import time

from keras.applications import xception, inception_v3, resnet50, vgg16, vgg19
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

from utils import ProcessBar


def extract_features_v4(image_path, save_path, model_type, test=False, batch_size=1000):
    """
    Extract Cat Vs Dog data features.

    Args:
        image_path: images path
        save_path: path to save features
        model_type: inception, xception, resnet50, vgg16 or vgg19
        test: If True, label for each data is 0/1,
              If False, label is the images' id
        batch_size: Each batch have how many data(4096 rows)
    """

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print('mkdir', save_path)

    # Keras Base model and its preprocess function.
    if model_type == 'inception':
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        preprocess = inception_v3.preprocess_input
        image_size = (299, 299)
    elif model_type == 'xception':
        base_model = xception.Xception(weights='imagenet', include_top=False)
        preprocess = xception.preprocess_input
        image_size = (299, 299)
    elif model_type == 'resnet50':
        base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
        preprocess = resnet50.preprocess_input
        image_size = (299, 299)
    elif model_type == 'vgg16':
        base_model = vgg16.VGG16(weights='imagenet', include_top=False)
        preprocess = vgg16.preprocess_input
        image_size = (224, 224)
    elif model_type == 'vgg19':
        base_model = vgg19.VGG19(weights='imagenet', include_top=False)
        preprocess = vgg19.preprocess_input
        image_size = (224, 224)
    else:
        raise ValueError('Unknown model type', model_type)

    # Add GlobalAveragePooling2D to reduce number of features.
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    features = []
    labels = []

    # Pre-shuffled train and test files.
    if not test:
        files = [os.path.join(image_path, f) for f in np.load('log/files_train.npy')]
    else:
        files = [os.path.join(image_path, f) for f in np.load('log/files_test.npy')]

    print('Begin to extracting {} images features...'.format(len(files)))
    start = time.time()
    bar = ProcessBar(len(files))
    current_batch = 0
    for i, file in enumerate(files):
        label = file.split('/')[-1].split('.')[0]
        if not test:
            label = 1 if label == 'dog' else 0
        else:
            label = int(label)

        # Below code come from Keras-document...
        img = image.load_img(file, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess(x)
        feature = model.predict(x)
        feature = np.reshape(feature, (-1))

        features.append(feature)
        labels.append(label)

        bar.show_process()

        if len(features) == batch_size or i == len(files) - 1:
            batch_path = os.path.join(save_path, 'batch_{}'.format(current_batch))
            if not os.path.isdir(batch_path):
                os.mkdir(batch_path)
            features_path = os.path.join(batch_path, 'features.npy')
            labels_path = os.path.join(batch_path, 'labels.npy')

            np.save(features_path, features)
            np.save(labels_path, labels)

            features = []
            labels = []

            current_batch += 1
    print('saved at', save_path)
    np.save(os.path.join(save_path, 'number.npy'), [current_batch + 1])
    print('Took {:.2f} seconds.'.format(time.time() - start))
