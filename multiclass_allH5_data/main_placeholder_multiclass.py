import os
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import config
import batch_inputs
import evaluation
import training
import inference_gray
inference = inference_gray


IMAGE_SIZE = 176

FLAGS = tf.app.flags.FLAGS


def train(is_finetune=False):

    tf.reset_default_graph()
    startstep = 0 if not is_finetune else int(FLAGS.finetune_dir.split('-')[-1])
    with tf.Graph().as_default():
        # ++++++++++++++++++++++++ TRAINING INPUT LAODING ++++++++++++++++++++++++
        x_train, y_train, id_train = batch_inputs.inputs(['./record/train.tfrecords'],
                                                         FLAGS.batch_size, False)
        # print(x_train.shape)
        y_train = tf.one_hot(y_train, FLAGS.num_class)
        # print(y_train.shape)
        tf.summary.image('images', x_train)
        x_train = tf.image.resize_image_with_crop_or_pad(x_train, IMAGE_SIZE,
                                                         IMAGE_SIZE)
        y_train = tf.image.resize_image_with_crop_or_pad(y_train, IMAGE_SIZE,
                                                         IMAGE_SIZE)
        # ++++++++++++++++++++++++ TESTING INPUT LAODING ++++++++++++++++++++++++
        x_test, y_test, id_test = batch_inputs.inputs(['./record/test.tfrecords'],
                                                      FLAGS.batch_size, True)
        y_test = tf.one_hot(y_test, FLAGS.num_class)
        tf.summary.image('images', x_test)
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

        loss = evaluation.loss_calc(logits=logits, labels=labels)
        train_op, global_step = training.training(loss=loss)
        accuracy = tf.argmax(logits, axis=3)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=1000)

        with tf.Session() as sess:

            if(is_finetune):
                print("\n =====================================================")
                print("  Finetuning with model: ", FLAGS.model)
                print("\n    Batch size is: ", FLAGS.batch_size)
                print("    ckpt files are saved to: ", FLAGS.log_dir)
                print("    Max iterations to train is: ", config.n_train_steps)
                print(" =====================================================")
                saver.restore(sess, FLAGS.finetune_dir)
            else:
                print("\n =====================================================")
                print("  Training from scratch with model: ", FLAGS.model)
                print("\n    Batch size is: ", FLAGS.batch_size)
                print("    ckpt files are saved to: ", FLAGS.log_dir)
                print("    Max iterations to train is: ", config.n_train_steps)
                print(" =====================================================")
                sess.run(tf.variables_initializer(tf.global_variables()))
                sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            for step in range(startstep + 1, startstep + config.n_train_steps + 1):
                images_batch, labels_batch = sess.run(fetches=[x_train, y_train])

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True,
                                   keep_prob: 0.5}

                start_time = time.time()

                _, train_loss_value, \
                    train_accuracy_value, \
                    train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)

                # Finding duration for training batch
                duration = time.time() - start_time

                if step % 10 == 0:  # Print info about training
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = float(duration)

                    print('\n--- Normal training ---')
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, train_loss_value,
                                         examples_per_sec, sec_per_batch))

                    # eval current training batch pre - class accuracy
                    pred = sess.run(logits, feed_dict=train_feed_dict)
                    # evaluation.per_class_acc(pred, labels_batch)  # printing class accuracy

                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()

                if step % 100 == 0 or (step + 1) == config.n_train_steps:
                    # test_iter = FLAGS.num_examples_epoch_test // FLAGS.test_batch_size
                    test_iter = FLAGS.num_examples_epoch_test // FLAGS.batch_size
                    """ Validate training by running validation dataset """
                    print("\n===========================================================")
                    print("--- Running test on VALIDATION dataset ---")
                    total_val_loss = 0.0
                    # hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
                    for val_step in range(test_iter):
                        test_img_batch, test_lbl_batch = sess.run(fetches=[x_test,
                                                                           y_test])

                        val_feed_dict = {images: test_img_batch,
                                         labels: test_lbl_batch,
                                         is_training: True,
                                         keep_prob: 1.0}

                        _val_loss, _val_pred = sess.run(fetches=[loss, logits],
                                                        feed_dict=val_feed_dict)
                        total_val_loss += _val_loss
                        # hist += evaluation.get_hist(_val_pred, val_labels_batch)
                    print("Validation Loss: ", total_val_loss / test_iter, ". If this value increases the model is likely overfitting.")
                    # evaluation.print_hist_summery(hist)
                    print("===========================================================")

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or step % 200 == 0 \
                        or (step + 1) == config.n_train_steps:
                    print("\n--- SAVING SESSION ---")
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("=========================")

            coord.request_stop()
            coord.join(threads)


def main(args):
    if FLAGS.testing:
        print("Testing the model!")
        # AirNet.test()
    elif FLAGS.finetune:
        train(is_finetune=True)
    else:
        train(is_finetune=False)


if __name__ == "__main__":
    tf.app.run()  # wrapper that handles flags parsing.
