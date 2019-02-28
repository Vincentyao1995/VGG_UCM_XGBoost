import tensorflow as tf
import numpy as np
import os
import config


# you need to change this to your data directory
train_dir = r'/media/jsl/ubuntu/data/UCMerced_LandUse/jpgImages/'
test_dir = r'/home/jsl/Desktop/data/UCMerced_LandUse/jpgImages/test/'


def get_files(data_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    imgs = []
    labels = []
    for folder in os.listdir(data_dir):
        folder_path = data_dir + os.sep + folder
        for file in os.listdir(folder_path):
            file_path = folder_path + os.sep + file
            imgs.append(file_path)
            labels.append(config.aid_class2label[folder])

    temp = np.array([imgs, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.one_hot(indices=label, depth=config.aid_n_class, on_value=1.0, off_value=0.0)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

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
    image = tf.image.per_image_standardization(image)

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





def test_get_batch():
    import matplotlib.pyplot as plt

    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 256
    IMG_H = 256

    image_list, label_list = get_files(train_dir)

    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 100:

                img, label = sess.run([image_batch, label_batch])

                # just test one batch
                for j in np.arange(BATCH_SIZE):
                    print(label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test_get_batch()