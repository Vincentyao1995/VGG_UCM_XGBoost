import tensorflow as tf
import tools
import VGG
import config
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)

    # img = tf.image.resize_image_with_crop_or_pad(img, config.IMG_W, config.IMG_H)
    img2 =tf.image.resize_nearest_neighbor([img],[config.IMG_H,config.IMG_W])
    # with tf.Session() as sess:
    #     mm2 = sess.run(img2,feed_dict={img_path:'hd_0578.jpg'})[0]
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
    ckpt = tf.train.get_checkpoint_state(config.ucm_checkpoint_path)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)
        i = 0
        with tf.Session() as sess:
            i = 0
            saver.restore(sess, ckpt.model_checkpoint_path)
            mm2 = sess.run(img2, feed_dict={img_path: 'hd_0613.jpg'})
            pre = sess.run(predict, feed_dict={x:mm2})
            print(pre)
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)

            # try:
            #     while not coord.should_stop() and i < 1:
            #         mm2 = sess.run(img2, feed_dict={img_path: 'hd_0578.jpg'})[0]
            #         path_test, img_test, label_test = sess.run([path_batch, image_batch, label_batch])
            #         true_label_test = sess.run(true_label)
            #         acc, pred_test = sess.run([accuracy, predict], feed_dict={x: img_test, y_: label_test})
            #         str_acc = class_folder + str(acc) + '\r\n'
            #         print(acc)
            #         i += 1
            #         for index in range(len(pred_test)):
            #             if true_label_test[index] != pred_test[index]:
            #                 str_acc += path_test[index].decode('utf-8') + ':' + config.get_class_name_by_index(
            #                     pred_test[index]) + '\r\n'
            # except tf.errors.OutOfRangeError:
            #     print('done!')
            # finally:
            #     coord.request_stop()
            # coord.join(threads)
            # with open(class_folder + '.txt', 'w') as fp:
            #     fp.write(str_acc)
