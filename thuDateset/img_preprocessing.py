# -*-coding:utf-8-*-
import os
import cv2
import sys
import shutil
import matplotlib.pyplot as plt


def move_imgs_2_class_folder(raw_folder_path, target_folder_path, class_name, start_index):
    log_str = ''
    for one_img_folder in os.listdir(raw_folder_path):
        one_img_folder_path = os.path.join(raw_folder_path, one_img_folder)
        for img_file in os.listdir(one_img_folder_path):
            if img_file[-3:] == 'tif':
                tif_file_path = os.path.join(one_img_folder_path, img_file)
                tfw_file_path = tif_file_path[:-3] + 'tfw'
                img = cv2.imread(tif_file_path)
                imh_h = img.shape[0]
                img_w = img.shape[1]
                if imh_h >= 256 and img_w > 256:
                    img = img[0:256, 0:256, :]
                    target_jpg_path = target_folder_path + os.sep + class_name + '_' + str(start_index) + '.jpg'
                    target_tfw_path = target_folder_path + os.sep + class_name + '_' + str(start_index) + '.tfw'
                    cv2.imwrite(target_jpg_path, img)
                    shutil.copy(tfw_file_path, target_tfw_path)
                    start_index += 1
                    log_str += tif_file_path + ',' + target_jpg_path + '\r\n'
    with open('log.txt', 'a') as f:
        f.write(log_str)


if __name__ == '__main__':
    test_raw_folder_path = r'/media/jsl/ubuntu/map/map3'
    test_target_folder_path = r'/media/jsl/ubuntu/residential'
    start_index = 153
    move_imgs_2_class_folder(test_raw_folder_path, test_target_folder_path, 'residential', start_index)
