# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import config
import numpy as np


def inputs(record_file, batch_size=32, do_test=False):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string),
               'train/id': tf.FixedLenFeature([], tf.string)}

    # filename_queue = tf.train.string_input_producer(record_file,
    #                                                 num_epochs=config.N_EPOCHS
    #                                                 if not do_test else 1)
    filename_queue = tf.train.string_input_producer(record_file,
                                                    num_epochs=config.N_EPOCHS)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['train/image'], tf.float32)
    # label = tf.decode_raw(features['train/label'], tf.float32)
    label = tf.decode_raw(features['train/label'], tf.int32)
    ID = features['train/id']

    image = tf.reshape(image, [config.ORIG_SIZE, config.ORIG_SIZE])
    label = tf.reshape(label, [config.ORIG_SIZE, config.ORIG_SIZE])

    min_fraction_of_examples_in_queue = 0.4
    num_examples_per_epoch = config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN if not do_test else config.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    num_preprocess_threads = 16
    images, labels, ID_batch = tf.train.shuffle_batch(
        [image, label, ID],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        allow_smaller_final_batch=True)

    return images, labels, ID_batch


if __name__ == '__main__':

    TEST_BATCH_SIZE = 10
    images, labels, ids = inputs(['./record/test.tfrecords'],
                                 TEST_BATCH_SIZE, True)

    labels_onehot = tf.one_hot(labels, 2)

    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, lbl, id, labels_onehot_val = sess.run([images, labels, ids, labels_onehot])

        # import matplotlib.pyplot as plt

        # print('Image : ')
        # print(img.shape)
        # plt.imshow(img[1, :, :])
        # plt.show()

        print('Label : ')
        print(lbl.shape)
        # plt.imshow(lbl[1, :, :])
        # plt.show()

        # # print('++++++++++')
        a = np.asarray(lbl[1, :, :], np.int32)
        print(np.unique(a))
        print(np.max(a))
        print(np.min(a))

        # print('ID : ')
        # print(id.shape)
        # print(id[1])

        print('One hot encoded labels : ')
        # print(labels_onehot_val)
        print(labels_onehot_val.shape)
        print(np.min(labels_onehot_val))
        print(np.max(labels_onehot_val))
        print(np.unique(labels_onehot_val))

        coord.request_stop()
        coord.join(threads)
        sess.close()
