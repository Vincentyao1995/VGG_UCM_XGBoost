import tensorflow as tf
import os
import config
import numpy as np
import matplotlib.pyplot as plt
import VGG
import tools
import cv2
import sys
import utils
from libtiff import TIFF


def get_one_class_files(class_folder_path):
    imgs = []
    labels = []

    for file in os.listdir(class_folder_path):
        file_path = os.path.join(class_folder_path, file)
        class_name = file_path.split('/')[-2]
        imgs.append(file_path)
        labels.append(config.class2label[class_name])

    temp = np.array([imgs, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_one_class_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.one_hot(indices=label, depth=21, on_value=1.0, off_value=0.0)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    # image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # you can also use shuffle_batch
    # image_batch, label_batch = tf.train.shuffle_batch([image, label],
    #                                                   batch_size=batch_size,
    #                                                   num_threads=64,
    #                                                   capacity=capacity,
    #                                                   min_after_dequeue=capacity - 1)

    # label_batch = tf.reshape(label_batch, [batch_size, 21])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


def get_one_hot_label(label):
    return tf.one_hot(indices=label, depth=config.n_class, on_value=1.0, off_value=0.0)


def get_one_class_batch_new(image, label, batch_size, image_H=config.img_Height, image_W=config.img_Width,
                            capacity=256):
    images_path = tf.cast(image, tf.string)
    label = tf.one_hot(indices=label, depth=21, on_value=1.0, off_value=0.0)

    # make an input queue
    input_queue = tf.train.slice_input_producer([images_path, label], shuffle=False)

    label = input_queue[1]
    path = input_queue[0]
    image_contents = tf.read_file(path)
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    # image = tf.image.per_image_standardization(image)

    path_batch, image_batch, label_batch = tf.train.batch([path, image, label],
                                                          batch_size=batch_size,
                                                          num_threads=64,
                                                          capacity=capacity)

    # you can also use shuffle_batch
    # image_batch, label_batch = tf.train.shuffle_batch([image, label],
    #                                                   batch_size=batch_size,
    #                                                   num_threads=64,
    #                                                   capacity=capacity,
    #                                                   min_after_dequeue=capacity - 1)

    # label_batch = tf.reshape(label_batch, [batch_size, 21])
    image_batch = tf.cast(image_batch, tf.float32)

    return path_batch, image_batch, label_batch

def test_rs_img(img = None):
    #this function is to testing rs image pixel-pixel mode classification. pixel_based_CNN used

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    img2 = tf.image.resize_nearest_neighbor([img], [config.img_Height, config.img_Width])

    x = tf.placeholder(tf.float32, shape=[config.test_batch_size, config.img_Width, config.img_Height, 3])
    y_ = tf.placeholder(tf.int16, shape=[config.test_batch_size, config.n_class])

    logits = VGG.VGG16N(x, config.n_class, False)

    predict = tf.argmax(logits, 1)
    # true_label = tf.argmax(label_batch, 1)
    # loss = tools.loss(logits, y_)
    # accuracy = tools.accuracy(logits, y_)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)

    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)

        with tf.Session() as sess:

            saver.restore(sess, ckpt.model_checkpoint_path)

            rsImg_img = img

            (rsImg_height, rsImg_width, rsImg_channels) = rsImg_img.shape

            num_hori_windows = int(rsImg_width/config.img_Width) + 1
            num_verti_windows = int(rsImg_height/config.img_Height) + 1
            for i in range(num_hori_windows):
                for j in range(num_verti_windows):

                    #deal with out of boundary situation.
                    left_boundary = i*config.img_Width
                    right_boundary = (i+1)*config.img_Width
                    top_boundary = j*config.img_Height
                    bottom_boundary = (j+1)*config.img_Height
                    if right_boundary > rsImg_width:
                        right_boundary = rsImg_width
                        left_boundary = rsImg_width - config.img_Width
                    if bottom_boundary > rsImg_height:
                        bottom_boundary = rsImg_height
                        top_boundary = rsImg_height - config.img_Height

                    rsImg_window = rsImg_img[left_boundary:right_boundary, top_boundary:bottom_boundary]
                    img_content = sess.run(img2, feed_dict={img_path: rsImg_window})
                    pre = sess.run(predict, feed_dict={x: img_content})
                    print('prediction:%d' % pre)

def test_dataset():
    test_img_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny'

    test_rs_img(test_img_path)

    class_folder = 'tenniscourt'
    class_folder_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/validation/' + class_folder
    test_img_list, test_label_list = get_one_class_files(class_folder_path)
    # test_img_batch, test_label_batch = get_one_class_batch_new(test_img_list, test_label_list,
    #                                                            config.img_Width,
    #                                                            config.img_Height,
    #                                                            config.test_batch_size, 256)
    path_batch, image_batch, label_batch = get_one_class_batch_new(test_img_list, test_label_list,
                                                                   config.test_batch_size, 256)
    x = tf.placeholder(tf.float32, shape=[config.test_batch_size, config.img_Width, config.img_Height, 3])
    y_ = tf.placeholder(tf.int16, shape=[config.test_batch_size, config.n_class])

    logits = VGG.VGG16N(x, config.n_class, False)

    predict = tf.argmax(logits, 1)
    true_label = tf.argmax(label_batch, 1)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)

    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)
        i = 0
        with tf.Session() as sess:
            i = 0
            saver.restore(sess, ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                while not coord.should_stop() and i < 1:
                    path_test, img_test, label_test = sess.run([path_batch, image_batch, label_batch])
                    true_label_test = sess.run(true_label)
                    acc, pred_test = sess.run([accuracy, predict], feed_dict={x: img_test, y_: label_test})
                    str_acc = class_folder + str(acc) + '\r\n'
                    print(acc)
                    i += 1
                    for index in range(len(pred_test)):
                        if true_label_test[index] != pred_test[index]:
                            str_acc += path_test[index].decode('utf-8') + ':' + config.get_class_name_by_index(
                                pred_test[index]) + '\r\n'
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
            with open(class_folder + '.txt', 'w') as fp:
                fp.write(str_acc)



if __name__ == '__main__':
    tif_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/obit/OBT-20181214-0016-sub.tif'
    jpg_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/obit/OBT-20181214-0016-sub_pca.tif'
    tif = TIFF.open(tif_path, mode='r')
    img = tif.read_image()
    img = img.astype(np.uint8)
    test_rs_img(img)

