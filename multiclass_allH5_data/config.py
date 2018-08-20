# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import


import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


SRC_NIFTY_DIR = './BraTS17/HGG/**/'
DST_JPG_DIR = './BRATS/HGG/'


MODALITY_DICT = {'flair': 0, 't1': 1, 't1s': 2, 't2': 3, 'gt': 4}
MODALITY = 't1'


IMG_MODE = 'reg'

TASK = 'all'


# For step2_write_tfrecord
TF_IMG_SRC = './BRATS/HGG/t1/reg_JPG/*'
# H5_SRC = './BRATS/HGG/HGG_patient_*.h5'
H5_SRC = './BRATS/HGG/'


""" AFFECTS HOW CODE RUNS"""

tf.app.flags.DEFINE_string('model', 'basic',
                           """ Defining what version of the model to run """)

# Training
tf.app.flags.DEFINE_string('log_dir', "./ckpt_dir/",  # Training is default on, unless testing or finetuning is set to "True"
                           """ dir to store training ckpt """)
# tf.app.flags.DEFINE_integer('max_steps', "60000",
#                             """ max_steps for training """)

# Testing
tf.app.flags.DEFINE_boolean('testing', False,  # True or False
                            """ Whether to run test or not """)
tf.app.flags.DEFINE_string('model_ckpt_dir', "./ckpt/model.ckpt-1800",
                           """ checkpoint file for model to use for testing """)
tf.app.flags.DEFINE_boolean('save_image', True,
                            """ Whether to save predicted image """)
tf.app.flags.DEFINE_string('res_output_dir', "./result_imgs",
                           """ Directory to save result images when running test """)
# Finetuning
tf.app.flags.DEFINE_boolean('finetune', False,  # True or False
                            """ Whether to finetune or not """)
tf.app.flags.DEFINE_string('finetune_dir', './ckpt/model.ckpt-1800',
                           """ Path to the checkpoint file to finetune from """)


""" TRAINING PARAMETERS"""
tf.app.flags.DEFINE_integer('batch_size', "64",
                            """ train batch_size """)
tf.app.flags.DEFINE_integer('test_batch_size', "1",
                            """ batch_size for training """)
tf.app.flags.DEFINE_integer('eval_batch_size', "6",
                            """ Eval batch_size """)

tf.app.flags.DEFINE_float('balance_weight_0', 0.8,
                          """ Define the dataset balance weight for class 0 - Not building """)
tf.app.flags.DEFINE_float('balance_weight_1', 1.1,
                          """ Define the dataset balance weight for class 1 - Building """)


""" DATASET SPECIFIC PARAMETERS """
# Directories
# tf.app.flags.DEFINE_string('train_dir', "../BRATS/HGG/",
#                            """ path to training images """)
# tf.app.flags.DEFINE_string('test_dir', "../BRATS/HGG/",
#                            """ path to test image """)
# tf.app.flags.DEFINE_string('val_dir', "../BRATS/HGG/",
#                            """ path to val image """)

# Dataset size. #Epoch = one pass of the whole dataset.
tf.app.flags.DEFINE_integer('num_epochs', "10000",
                            """ num of epochs on train training """)
tf.app.flags.DEFINE_integer('num_examples_epoch_train', "3500",
                            """ num examples per epoch for train """)
tf.app.flags.DEFINE_integer('num_examples_epoch_test', "500",
                            """ num examples per epoch for test """)
tf.app.flags.DEFINE_integer('num_examples_epoch_val', "50",
                            """ num examples per epoch for test """)
tf.app.flags.DEFINE_float('fraction_of_examples_in_queue', "0.1",
                          """ Fraction of examples from datasat to put in queue. Large datasets need smaller value, otherwise memory gets full. """)

# Image size and classes
tf.app.flags.DEFINE_integer('image_h', "176",
                            """ image height """)
tf.app.flags.DEFINE_integer('image_w', "176",
                            """ image width """)
tf.app.flags.DEFINE_integer('image_c', "4",
                            """ number of image channels (RGB) (the depth) """)
tf.app.flags.DEFINE_integer('num_class', "4",  # classes are "Building" and "Not building"
                            """ total class number """)


# FOR TESTING:
TEST_ITER = FLAGS.num_examples_epoch_test // FLAGS.batch_size


tf.app.flags.DEFINE_float('moving_average_decay', "0.99",  # "0.9999", #https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
                          """ The decay to use for the moving average""")


if(FLAGS.model == "basic" or FLAGS.model == "basic_dropout"):
    tf.app.flags.DEFINE_string('conv_init', 'xavier',  # xavier / var_scale
                               """ Initializer for the convolutional layers. One of: "xavier", "var_scale".  """)
    tf.app.flags.DEFINE_string('optimizer', "SGD",
                               """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)


train_n_batches = int(FLAGS.num_examples_epoch_train / FLAGS.batch_size)
test_n_batches = int(FLAGS.num_examples_epoch_test / FLAGS.test_batch_size)
n_train_steps = FLAGS.num_epochs * train_n_batches


NUM_EXAMPLES_PER_EPOCH_FOR_TEST = FLAGS.num_examples_epoch_test
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_epoch_train

# FLAGS.image_h
ORIG_SIZE = 184
IMAGE_SIZE = 184
N_EPOCHS = FLAGS.num_epochs
TEST_SPLIT = 0.2
