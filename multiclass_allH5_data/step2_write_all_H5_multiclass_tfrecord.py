# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import sys
import numpy as np
import cv2
import os
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

    writer.close()
    sys.stdout.flush()


def load_data(path, do_test=False):
    images = []
    labels = []
    ids = []
    print('Reading images')

    mod_dict = dict((v, k) for k, v in config.MODALITY_DICT.iteritems())

    for i in os.listdir(path):
        tmp_list = i.split('_')
        patient_num = tmp_list[2]
        slice_ix = tmp_list[3]

        h5f = h5py.File(os.path.join(path, i), 'r')

        # +++++++++++++++++++++++++ IMAGE +++++++++++++++++++++++++
        mod_images = []
        for mod in range(4):
            dataset_name = '{}_{}_{}'.format(mod_dict[mod],
                                             patient_num, slice_ix)
            img = h5f[dataset_name][:]
            mod_images.append(img)

        images.append(mod_images)

        # +++++++++++++++++++++++++ LABEL +++++++++++++++++++++++++
        lbl = h5f['gt_{}_{}'.format(patient_num, slice_ix)][:]
        labels.append(lbl)

        h5f.close()

        # +++++++++++++++++++++++++++ ID ++++++++++++++++++++++++++
        ids.append(i)

    images = np.array(images, dtype=np.float32)
    images = images.transpose((0, 2, 3, 1))
    labels = np.array(labels, dtype=np.int32)
    ids = np.array(ids)

    print('images shape : ', images.shape)
    print('labels shape : ', labels.shape)

    return images, labels, ids


def creat_tf_records():

    images_data, labels_data, ids_data = load_data(config.H5_SRC)
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
