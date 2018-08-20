import os
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import config
import batch_inputs
import evaluation
import inference_gray
inference = inference_gray


IMAGE_SIZE = 176

FLAGS = tf.app.flags.FLAGS


def test():

    tf.reset_default_graph()
    with tf.Graph().as_default():

        # ++++++++++++++++++++++++ TESTING INPUT LAODING ++++++++++++++++++++++++
        x_test, y_test, id_test = batch_inputs.inputs(['./record/test.tfrecords'],
                                                      FLAGS.batch_size, True)
        y_test = tf.one_hot(y_test, FLAGS.num_class)
        x_test = tf.image.resize_image_with_crop_or_pad(x_test, IMAGE_SIZE,
                                                        IMAGE_SIZE)
        y_test = tf.image.resize_image_with_crop_or_pad(y_test, IMAGE_SIZE,
                                                        IMAGE_SIZE)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")
        images = tf.placeholder(tf.float32,
                                shape=[None,
                                       FLAGS.image_h, FLAGS.image_w, FLAGS.image_c])
        labels = tf.placeholder(tf.int64,
                                [None,
                                 FLAGS.image_h, FLAGS.image_w, FLAGS.num_class])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print('++++++++ Mode building starts here +++++++++')
        if FLAGS.model == "basic":
            logits = inference.inference_basic(images, is_training)
        elif FLAGS.model == "extended":
            logits = inference.inference_extended(images, is_training)
        elif FLAGS.model == "basic_dropout":
            logits = inference.inference_basic_dropout(images, is_training, keep_prob)
        elif FLAGS.model == "extended_dropout":
            logits = inference.inference_extended_dropout(images, is_training, keep_prob)
        else:
            raise ValueError("The selected model does not exist")

        sfm_logits = tf.nn.softmax(logits)
        class_pred = tf.argmax(logits, axis=3)
        y_test_argmax = tf.argmax(y_test, axis=3)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print tf.train.latest_checkpoint(FLAGS.log_dir)
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

            # sess.run(tf.variables_initializer(tf.global_variables()))
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            test_img_batch, test_lbl_batch, \
                test_id_batch, y_test_argmax_val = sess.run(fetches=[x_test,
                                                                     y_test,
                                                                     id_test,
                                                                     y_test_argmax])
            print(test_lbl_batch.shape)

            test_feed_dict = {images: test_img_batch,
                              labels: test_lbl_batch,
                              is_training: True,
                              keep_prob: 1.0}

            class_pred_val, sfm_logits_val, pred = sess.run([class_pred, sfm_logits, logits], feed_dict=test_feed_dict)

            print('SFM Logits Shape : ', sfm_logits_val.shape)
            print('Unique values in SFM Logits : ', np.unique(sfm_logits_val))
            print('Predicted Class Label Shape : ', class_pred_val.shape)

            #
            import h5py

            h5f = h5py.File('results.h5', 'w')

            for i in range(FLAGS.batch_size):
                img = test_img_batch[i, :, :, :]
                lbl = y_test_argmax_val[i, :, :]
                pred_img = class_pred_val[i, :, :]
                ID = test_id_batch[i]
                h5f.create_dataset('image_{}'.format(i), data=img)
                h5f.create_dataset('label_{}'.format(i), data=lbl)
                h5f.create_dataset('pred_{}'.format(i), data=pred_img)
                h5f.create_dataset('id_{}'.format(i), data=ID)

            #

            h5f.close()
            print('H5 file written !!')

            coord.request_stop()
            coord.join(threads)


def main(args):
    if True:
        print("Testing the model!")
        test()


if __name__ == "__main__":
    tf.app.run()  # wrapper that handles flags parsing.
