'''
this file contain a funtion which uses pretrained VGG16 net to extract features from images under dataset_root_path, written output into .txt file under dataset_feature_root_path. 
'''

import VGG
import tensorflow as tf
import config
import matplotlib.pyplot as plt
import tools
import os

def ext_dataset_fea_use_vgg16(dataset_root_path, dataset_feature_root_path):
    pre_trained_weights = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/train/vgg16.npy'
    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    img = tf.image.resize_image_with_crop_or_pad(img, config.IMG_W, config.IMG_H)
    img = tf.image.per_image_standardization(img)
    # with tf.Session() as sess:
    #     image = sess.run(img, feed_dict={img_path:'agricultural00.jpg'})
    #     plt.imshow(image)
    #     plt.show()
    x = tf.placeholder(tf.float32, shape=[1, config.IMG_W, config.IMG_H, 3])
    # y_ = tf.placeholder(tf.int16, shape=[1, config.N_CLASSES])
    #
    img_fea = VGG.VGG16N_CNN(x, config.N_CLASSES, False)
    with tf.Session() as sess:
        tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])
        for class_name in os.listdir(dataset_root_path):
            class_path = os.path.join(dataset_root_path, class_name)
            print('--**extracting %s**--' % class_path)
            fea_class_path = os.path.join(dataset_feature_root_path, class_name)
            if not os.path.exists(fea_class_path):
                os.mkdir(fea_class_path)
            for img_name in os.listdir(class_path):

                jpg_img_path = os.path.join(class_path, img_name)
                fea_txt_path = os.path.join(fea_class_path, img_name[:-4]+'.txt')
                print('extracting %s' % jpg_img_path)

                image = sess.run(img, feed_dict={img_path: jpg_img_path})
                # plt.imshow(image)
                # plt.show()
                img_fea_out = sess.run(img_fea, feed_dict={x: [image]})
                # print(img_fea_out.shape)

                str_img_fea = ','.join(map(str, img_fea_out[0].tolist()))
                with open(fea_txt_path, 'w') as f:
                    f.write(str_img_fea)




if __name__ == '__main__':
    print('extracting feature using vgg16-ucm....')
    ucm_jpg_root_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/train'
    ucm_fea_root_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/fea_extract'
    ext_dataset_fea_use_vgg16(ucm_jpg_root_path, ucm_fea_root_path)
    print('extraction done! feature saved in ~/fea_extract ')