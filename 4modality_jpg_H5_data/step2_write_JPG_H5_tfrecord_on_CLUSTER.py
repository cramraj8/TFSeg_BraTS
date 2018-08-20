# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import sys
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


import config
import h5py


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(imgs, lbls, IDs, tfrecord_name='./train.tfrecords', lbl='train'):

    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    n_obs = imgs.shape[0]
    for i in range(n_obs):
        if not i % 100:
            print('{} data: {}/{}'.format(lbl, i, n_obs))
            sys.stdout.flush()

        # Load the image
        img = imgs[i, :, :, :]
        lbl = lbls[i, :, :]
        ID = IDs[i]

        # Create a feature
        feature = {
            'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
            'train/label': _bytes_feature(tf.compat.as_bytes(lbl.tostring())),
            'train/id': _bytes_feature(tf.compat.as_bytes(ID))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
        # print('done')

    writer.close()
    sys.stdout.flush()


def binarize_targets(y_train):
    # y_train = np.asarray(y_train)

    task = 'all'

    if task == 'all':
        y_train = (y_train > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    # print('uniques elements in y_Train : ', np.unique(y_train))

    return y_train


def load_data(path, do_test=False):
    images = []
    labels = []
    ids = []

    print('Reading images')

    files = glob.glob(path)
    files.sort()
    for fl in files:

        mod_dict = dict((v, k) for k, v in config.MODALITY_DICT.iteritems())
        tmp_name_list = fl.split('/')
        mod_images = []
        for j in range(4):
            tmp_name_list[3] = mod_dict[j]
            t_fl = "/".join(tmp_name_list)

            # +++++++++++++++++++++++++ IMAGE +++++++++++++++++++++++++
            image = cv2.imread(t_fl)
            image = cv2.resize(image,
                               (config.ORIG_SIZE, config.ORIG_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mod_images.append(gray_image)

        images.append(mod_images)

        # +++++++++++++++++++++++++ LABEL +++++++++++++++++++++++++
        tmp_name_list[3] = 'gt'
        gt_fl = "/".join(tmp_name_list)
        gt_fl = list(gt_fl)
        gt_fl[-3:] = 'h5'
        gt_fl = "".join(gt_fl)

        h5f = h5py.File(gt_fl, 'r')
        lbl = h5f['gt'][:]
        lbl = binarize_targets(lbl)
        labels.append(lbl)
        h5f.close()

        flbase = os.path.basename(fl)
        ids.append(flbase)

    images = np.array(images, dtype=np.float32)
    images = images.transpose((0, 2, 3, 1))
    labels = np.array(labels, dtype=np.int32)
    ids = np.array(ids)

    print('images shape : ', images.shape)

    return images, labels, ids


def creat_tf_records():

    images_data, labels_data, ids_data = load_data(config.TF_IMG_SRC)
    print('Data Loaded.')
    print(' Data : ', images_data.shape, '\n')

    train_images, test_images, train_labels, test_labels, \
        train_ids, test_ids = train_test_split(images_data, labels_data, ids_data,
                                               test_size=config.TEST_SPLIT,
                                               random_state=42)

    print('Train data : ')
    print(train_images.shape)
    print(train_labels.shape)
    print(train_ids.shape)
    print(test_images.shape)
    print(test_labels.shape)
    print(test_ids.shape)
    print('++++++++++++++++++++++++++++++++')

    # ========================================
    # Shuffle
    train_images, train_labels, train_ids = shuffle(train_images, train_labels, train_ids)
    test_images, test_labels, test_ids = shuffle(test_images, test_labels, test_ids)

    TFRECORD_ROOT = './record/'
    if not os.path.exists(TFRECORD_ROOT):
        os.makedirs(TFRECORD_ROOT)

    # Write Train TFRecords
    write_record(train_images, train_labels, train_ids,
                 tfrecord_name=TFRECORD_ROOT + 'train.tfrecords', lbl='train')
    print('\n')
    # Write Test TFRecords
    write_record(test_images, test_labels, test_ids,
                 tfrecord_name=TFRECORD_ROOT + 'test.tfrecords', lbl='test')


if __name__ == '__main__':
    creat_tf_records()
