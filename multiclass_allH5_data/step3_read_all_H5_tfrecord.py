# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os

import config


def inputs(record_file, batch_size=32, do_test=False):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string),
               'train/id': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(record_file,
                                                    num_epochs=config.N_EPOCHS
                                                    if not do_test else 1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    # label = tf.decode_raw(features['train/label'], tf.float32)
    label = tf.decode_raw(features['train/label'], tf.int32)
    ID = features['train/id']

    image = tf.reshape(image, [config.ORIG_SIZE, config.ORIG_SIZE, 4])
    label = tf.reshape(label, [config.ORIG_SIZE, config.ORIG_SIZE])

    # Ensure that the random shuffling has good mixing properties.
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

    tf.summary.image('images', images)
    return images, labels, ID_batch


if __name__ == '__main__':

    TEST_BATCH_SIZE = 32
    images, labels, ids = inputs(['./record/train.tfrecords'],
                                 TEST_BATCH_SIZE, True)

    import matplotlib.pyplot as plt

    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, lbl, id = sess.run([images, labels, ids])

        INDEX = 20

        print('Image : ')
        print(img.shape)
        for i in range(4):
            plt.imshow(img[INDEX, :, :, i])
            plt.show()

        print('Label : ')
        print(lbl.shape)
        plt.imshow(lbl[INDEX, :, :])
        plt.show()

        import numpy as np
        print('++++++++++')
        a = np.asarray(lbl[INDEX, :, :], np.float32)
        # a = np.asarray(lbl[1, :, :], np.int32)
        print(np.unique(a))
        print(np.max(a))
        print(np.min(a))

        print('ID : ')
        print(id.shape)
        print(id[INDEX])

        coord.request_stop()
        coord.join(threads)
        sess.close()
