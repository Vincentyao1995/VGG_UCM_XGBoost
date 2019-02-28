#this file has been changed by vincent in 2019/01/25
#this file is to separate training set into validation and testing

import cv2
import os
import random


def convert_tif2jpg(tif_folder_path, jpg_folder_path):
    if not os.path.exists(jpg_folder_path):
        os.mkdir(jpg_folder_path)
        print('qq')
    for class_folder in os.listdir(tif_folder_path):
        jpg_class_folder = jpg_folder_path + class_folder
        if not os.path.exists(jpg_class_folder):
            os.mkdir(jpg_class_folder)
        for tif_img in os.listdir(tif_folder_path + class_folder):
            tif_img_path = tif_folder_path + class_folder + os.sep + tif_img
            jpg_img_path = jpg_folder_path + class_folder + os.sep + tif_img[:-3] + 'jpg'
            img = cv2.imread(tif_img_path)
            cv2.imwrite(jpg_img_path, img)
            print(jpg_img_path)


def rotate_img(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_dataset(root_path, angle, suffix):
    # this function is to rotate dataset to create larger dataset to train.
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for folder in os.listdir(root_path):
        class_folder_path = root_path + os.sep + folder
        for img in os.listdir(class_folder_path):
            img_path = class_folder_path + os.sep + img
            new_img_path = img_path[:-4] + '_' + suffix + '.jpg'
            img_matrix = cv2.imread(img_path)
            img_matrix_rotate = rotate_img(img_matrix, angle)
            cv2.imwrite(new_img_path, img_matrix_rotate)


def flip_img(image, horizontal=1):
    img_flip = cv2.flip(image, 1)
    return img_flip


def flip_dataset(root_path):
    for folder in os.listdir(root_path):
        class_folder_path = root_path + os.sep + folder
        for img in os.listdir(class_folder_path):
            img_path = class_folder_path + os.sep + img
            new_img_path = img_path[:-4] + '_flip.jpg'
            img_matrix = cv2.imread(img_path)
            img_matrix_flip = flip_img(img_matrix)
            cv2.imwrite(new_img_path, img_matrix_flip)


def split_train_test(train_dir_parent):
    #this function is to split training data to validation data and test data, into the parent folder of training then create validation, test folder automatically.
    # train_dir = r'/media/jsl/ubuntu/UCMerced_LandUse/jpgImages_rotate/train/'
    # validation_dir = r'/media/jsl/ubuntu/UCMerced_LandUse/jpgImages_rotate/validation/'
    # test_dir = r'/media/jsl/ubuntu/UCMerced_LandUse/jpgImages_rotate/test/'
    train_dir = train_dir_parent + os.sep + 'train' + os.sep
    validation_dir = train_dir_parent + os.sep + 'validation' + os.sep
    test_dir = train_dir_parent + os.sep + 'test' + os.sep
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for folder in os.listdir(train_dir):
        train_folder_path = train_dir + folder
        validation_folder_path = validation_dir + folder
        test_folder_path = test_dir + folder
        if not os.path.exists(test_folder_path):
            os.mkdir(test_folder_path)
        if not os.path.exists(validation_folder_path):
            os.mkdir(validation_folder_path)
        list_train_img = []
        for img in os.listdir(train_folder_path):
            img_train_path = train_folder_path + os.sep + img
            list_train_img.append(img_train_path)

        #here is to set how many validation/test images should be set.
        num_validation_img = int(len(list_train_img)*0.3)
        list_validation_img = random.sample(list_train_img, num_validation_img)
        for img_train in list_validation_img:
            img_file = cv2.imread(img_train)
            #attention, to promise no bug, make sure your folderpath don't contrain 'train' except train file at the same level with 'validation' and 'test'
            img_validation = img_train.replace('train', 'validation')
            cv2.imwrite(img_validation, img_file)

            os.remove(img_train)
            print(img_train)
            print(img_validation)

'''
        list_test_img = random.sample(list_train_img, 100)
        for img_train in list_test_img:
            img_file = cv2.imread(img_train)
            #attention, to promise no bug, make sure your folderpath don't contrain 'train' except train file at the same level with 'validation' and 'test'
            img_test = img_train.replace('train', 'test')
            cv2.imwrite(img_test, img_file)

            os.remove(img_train)
            print(img_train)
            print(img_validation)
'''
if __name__ == '__main__':
    # tif_folder_path = r'/media/jsl/ubuntu/UCMerced_LandUse/Images/'
    # jpg_folder_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM'
    jpg_folder_rotate_path = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/train'
    train_dir_parent = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset'

#double rotate could produce all four direction images.
    #rotate_dataset(jpg_folder_rotate_path,90,'1')
    #rotate_dataset(jpg_folder_rotate_path, 180, '2')

    split_train_test(train_dir_parent)

    # convert_tif2jpg(tif_folder_path, jpg_folder_path)
    test = r'/media/jsl/ubuntu/test'
    # rotate_dataset(jpg_folder_rotate_path, 90, '1')
    # rotate_dataset(jpg_folder_rotate_path, 180, '2')
    # flip_dataset(jpg_folder_rotete_path)
    # split_train_test()
