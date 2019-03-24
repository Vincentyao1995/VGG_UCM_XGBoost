import tensorflow as tf
import tools
import VGG
import config
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
sys.path.append('./')
from input_data import get_batch, get_files
import time
import utils

BATCH_SIZE = 16
learning_rate = 0.001
MAX_STEP = 15000  # it took me about one hour to complete the training.
IS_PRETRAIN = True
CAPACITY = 256

dataset_config = {
    'nwpu':
        {'checkpoint_path': config.nwpu_checkpoint_path,
         'n_class': config.nwpu_n_class,
         'data_path': config.nwpu_data_path,
         'class2label': config.nwpu_class2label
         },
    'ucm':
        {'checkpoint_path': config.ucm_checkpoint_path,
         'n_class': config.ucm_n_class,
         'data_path': config.ucm_data_path,
         'class2label': config.ucm_class2label
         },
    'thu':
        {'checkpoint_path': config.aid_checkpoint_path,
         'n_class': config.aid_n_class,
         'data_path': config.aid_data_root_path,
         'class2label': config.aid_class2label
         }}

# %%   Training

def train_aid():
    pre_trained_weights = r'/media/jsl/ubuntu/pretrain_weight/vgg16.npy'
    data_train_dir = os.path.join(config.aid_data_root_path, 'train')
    data_test_dir = os.path.join(config.aid_data_root_path, 'val')
    train_log_dir = os.path.join(config.aid_log_root_path, 'train')
    val_log_dir = os.path.join(config.aid_log_root_path, 'val')

    with tf.name_scope('input'):
        image_train_list, label_train_list = get_files(data_train_dir)
        image_val_list, label_val_list = get_files(data_test_dir)
        image_batch, label_batch = get_batch(image_train_list, label_train_list, config.aid_img_weight, config.aid_img_height, BATCH_SIZE, CAPACITY)
        val_image_batch, val_label_batch = get_batch(image_val_list, label_val_list, config.aid_img_weight, config.aid_img_height, BATCH_SIZE, CAPACITY)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, config.aid_img_weight, config.aid_img_height, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, config.aid_n_class])

    logits = VGG.VGG16N(x, config.aid_n_class, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print('start_time:', start_time)

    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([image_batch, label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
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

    coord.join(threads)
    sess.close()
    end_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    print('end_time:', end_time)

def test_vgg_single_img(image_path):

    global dataset_config

    dataset = dataset_config['ucm']

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    # img = tf.image.resize_image_with_crop_or_pad(img, config.IMG_W, config.IMG_H)
    img2 = tf.image.resize_nearest_neighbor([img], [config.IMG_H, config.IMG_W])

    x = tf.placeholder(tf.float32, shape=[1, config.IMG_W, config.IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[1, config.N_CLASSES])

    logits = VGG.VGG16N(x, config.N_CLASSES, False)

    predict = tf.argmax(logits, 1)
    # true_label = tf.argmax(label_batch, 1)
    # loss = tools.loss(logits, y_)
    # accuracy = tools.accuracy(logits, y_)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(dataset['checkpoint_path'])
    matrix_confusion = np.zeros((dataset['n_class'], dataset['n_class']))
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)
        i = 0
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            img_content = sess.run(img2, feed_dict={img_path: image_path})
            pre = sess.run(predict, feed_dict={x: img_content})
            print('prediction:%d' % pre)


def test_vgg_dataset():
    global dataset_config

    dataset = dataset_config['ucm']

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    # img = tf.image.resize_image_with_crop_or_pad(img, config.IMG_W, config.IMG_H)
    img2 = tf.image.resize_nearest_neighbor([img], [config.IMG_H, config.IMG_W])
    # with tf.Session() as sess:
    #     mm2 = sess.run(img2,feed_dict={img_path:'hd_0613.jpg'})[0]
    #     print(mm2.shape)
    #     plt.imshow(mm2)
    #
    #     plt.show()
    x = tf.placeholder(tf.float32, shape=[1, config.IMG_W, config.IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[1, config.N_CLASSES])

    logits = VGG.VGG16N(x, config.N_CLASSES, False)

    predict = tf.argmax(logits, 1)
    # true_label = tf.argmax(label_batch, 1)
    # loss = tools.loss(logits, y_)
    # accuracy = tools.accuracy(logits, y_)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(dataset['checkpoint_path'])
    matrix_confusion = np.zeros((dataset['n_class'], dataset['n_class']))
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)
        i = 0
        with tf.Session() as sess:
            i = 0
            saver.restore(sess, ckpt.model_checkpoint_path)
            val_data_path = os.path.join(dataset['data_path'], 'validation')
            for val_class_name in os.listdir(val_data_path):
                class_path = os.path.join(val_data_path, val_class_name)
                class_index = dataset['class2label'][val_class_name]
                for val_img_name in os.listdir(class_path):
                    val_img_path = os.path.join(class_path, val_img_name)
                    img_content = sess.run(img2, feed_dict={img_path: val_img_path})
                    pre = sess.run(predict, feed_dict={x: img_content})
                    print(class_index, pre)
                    matrix_confusion[class_index][pre] += 1

        utils.plot_confusion_matrix(matrix_confusion,
                                    normalize=False,
                                    target_names=config.ucm_class,
                                    title="Confusion Matrix")
        np.savetxt('ucm_vgg_confusion_matrix', matrix_confusion)

if __name__ == "__main__":
    img_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/validation/chaparral/chaparral03.jpg'
    test_vgg_single_img(img_path)




