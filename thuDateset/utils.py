import cv2
import gdal
import matplotlib.pyplot as plt
from libtiff import TIFF
import numpy as np
import config
import os
import shutil


def convert_img_tif2jpg(tif_path, jpg_path):
    # img = cv2.imread(tif_path)
    # cv2.imwrite(jpg_path, img)

    # dataset = gdal.Open(tif_path)
    # width = dataset.RasterXSize
    # height = dataset.RasterYSize
    # data = dataset.ReadAsArray(0, 0, width, height)
    # driver = gdal.GetDriverByName("GTiff")
    # driver.CreateCopy(jpg_path, dataset, 0, ["INTERLEAVE=PIXEL"])
    tif = TIFF.open(tif_path, mode='r')
    img = tif.read_image()
    img = img.astype(np.uint8)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(jpg_path, img)


def print_class2lable(data_train_path):
    class_names = []
    for class_name in os.listdir(data_train_path):
        class_names.append(class_name)
    class_names.sort()
    index = 0
    for class_name in class_names:
        print("'", class_name, "'", ':', index, ',')
        index += 1


def split_aid():
    aid_train_path = os.path.join(config.aid_data_root_path, 'train')
    aid_val_path = os.path.join(config.aid_data_root_path, 'val')
    for class_name in os.listdir(aid_train_path):
        train_class_path = os.path.join(aid_train_path, class_name)
        val_class_path = os.path.join(aid_val_path, class_name)
        if not os.path.exists(val_class_path):
            os.mkdir(val_class_path)
        for img_name in os.listdir(train_class_path):
            img_index = int(img_name.split('_')[1].split('.')[0])
            if img_index <= 80:
                train_img_path = os.path.join(train_class_path, img_name)
                val_img_path = os.path.join(val_class_path, img_name)
                shutil.move(train_img_path, val_img_path)


def rotate_img(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_aid():
    for data_type in os.listdir(config.aid_data_root_path):
        data_type_path = os.path.join(config.aid_data_root_path, data_type)
        for class_name in os.listdir(data_type_path):
            class_path = os.path.join(data_type_path, class_name)
            for img_name in os.listdir(class_path):
                src_img_path = os.path.join(class_path, img_name)
                tar_img_path = os.path.join(class_path, img_name.split('.')[0] + '_r.jpg')
                img_matrix = cv2.imread(src_img_path)
                img_matrix_rotate = rotate_img(img_matrix, 90)
                cv2.imwrite(tar_img_path, img_matrix_rotate)


def draw_Relu():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    x = np.arange(-10, 10)
    y = np.where(x < 0, 0, x)

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_yticks([-10, -5, 5, 10])

    plt.plot(x, y, label="ReLU", color="blue")
    plt.legend()
    plt.show()


def convert_img_png2jpg(png_path, jpg_path):
    img = cv2.imread(png_path)
    cv2.imwrite(jpg_path, img)


def rename_file_by_folder_name(folder_path):
    folder_name = folder_path.split("/")[-1]
    count = 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, folder_name + "_" + str(count) + '.jpg')
        count += 1
        print(file_path)
        print(new_file_path)
        os.rename(file_path, new_file_path)


def split_thudataset(train_folder, val_folder):
    for class_type_train in os.listdir(train_folder):
        class_path_train = os.path.join(train_folder, class_type_train)
        class_path_val = os.path.join(val_folder, class_type_train)
        if not os.path.exists(class_path_val):
            os.mkdir(class_path_val)
        for img in os.listdir(class_path_train):

            index = img.split(".")[0].split("_")[-1]
            print(img)
            index = int(index)
            if index > 500:
                train_img_path = os.path.join(class_path_train, img)
                val_img_path = os.path.join(class_path_val, img)
                shutil.move(train_img_path, val_img_path)


if __name__ == '__main__':
    # tif_path = r'/media/jsl/ubuntu/test/hd_001.tif'
    # jpg_path = r'/media/jsl/ubuntu/test/33.jpg'
    # convert_img_tif2jpg(tif_path, jpg_path)
    # nwpu_train_path = os.path.join(config.aid_data_root_path, 'train')
    # print_class2lable(nwpu_train_path)
    # split_aid()
    # rotate_aid()

    train_folder_path = r"/media/jsl/ubuntu/data/THUDataset"
    val_folder_path = r"/media/jsl/ubuntu/data/val"
    split_thudataset(train_folder_path, val_folder_path)

    # folder_path = r"/media/jsl/ubuntu/data/THUDataset/unuse/unuse__"
    # folder_path_new = r"/media/jsl/ubuntu/data/THUDataset/unuse/unuse_"
    #
    # for index in range(306, 612):
    #     img_path = folder_path + str(index) + ".jpg"
    #     print(img_path)
    #     img_path_new = folder_path_new + str(index) + ".jpg"
    #     os.rename(img_path, img_path_new)
