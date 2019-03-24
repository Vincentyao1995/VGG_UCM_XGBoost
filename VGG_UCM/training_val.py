# coding=UTF-8
# %%
# DATA:
# 1. cifar10(binary version):https://www.cs.toronto.edu/~kriz/cifar.html
# 2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
 
# TO Train and test:
# 0. get data ready, get paths ready !!!
# 1. run training_and_val.py and call train() in the console
# 2. call evaluate() in the console to test

# %%


import os
import os.path

import numpy as np
import tensorflow as tf

from input_data import get_files, get_batch, get_batch_datasetVersion
import VGG
import tools
import input_data

# %%
IMG_W = 256
IMG_H = 256
N_CLASSES = 21
BATCH_SIZE = 16
learning_rate = 0.001
MAX_STEP = 15000  # it took me about one hour to complete the training. Step is iteration
IS_PRETRAIN = True
CAPACITY = 256
RESTORE_MODEL = True


# %%   Training
def train():
    pre_trained_weights = r'/home/vincent/Desktop/jsl thesis/grad thesis/data/vgg16_pretrained/vgg16.npy'
    data_train_dir = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/train'
    data_test_dir = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/validation/'
    train_log_dir = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/train'
    val_log_dir = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/val'

    with tf.name_scope('input'):
        # tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=True,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=True)
        # val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=False,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=False)
        image_train_list, label_train_list = get_files(data_train_dir)
        image_val_list, label_val_list = get_files(data_test_dir)
        # image_batch, label_batch = get_batch(image_train_list, label_train_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        # val_image_batch, val_label_batch = get_batch(image_val_list, label_val_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        image_batch = get_batch_datasetVersion(image_train_list, label_train_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        val_batch = get_batch_datasetVersion(image_val_list, label_val_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    # x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    # y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[None, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])
    coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    #restore older checkpoints
    if RESTORE_MODEL == True:

        print("Reading checkpoints.../n")

        log_dir = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/train'
        model_name = r'model.ckpt-2000.meta'
        data_name = r'model.ckpt-2000'
        #restore Graph
        saver = tf.train.import_meta_graph(log_dir +os.sep + model_name)
        #restore paras
        saver.restore(sess, log_dir + os.sep + data_name)
        print("Loading checkpoints successfully!! /n")

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            #tra_images, tra_labels = sess.run([image_batch, label_batch])
            tra_images, tra_labels = sess.run(image_batch)
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run(val_batch)
                #val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    #coord.join(threads)
    sess.close()


# %%   Test the accuracy on test dataset. got about 85.69% accuracy.
import math


def evaluate():
    with tf.Graph().as_default():

        #        log_dir = 'C://Users//kevin//Documents//tensorflow//VGG//logsvgg//train//'
        log_dir = 'C:/Users/kevin/Documents/tensorflow/VGG/logs/train/'
        test_dir = './/data//cifar-10-batches-bin//'
        n_test = 10000

        images, labels = input_data.read_cifar10(data_dir=test_dir,
                                                 is_train=False,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)

        logits = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / BATCH_SIZE))
                num_sample = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

# %%

if __name__ == '__main__':

    #calc running time
    import time
    time_start = time.time()

    train()
    time_end = time.time()
    elapsed = time_end - time_start
    print('\n\ntime taken:',elapsed,'seconds.\n')

