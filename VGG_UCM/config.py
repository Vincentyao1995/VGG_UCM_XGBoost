import os
'''def convert_format(): use cv2, too slow to download, ignore temporarily'''
#import cv2
import random

class2label = {'agricultural': 0,
               'airplane': 1,
               'baseballdiamond': 2,
               'beach': 3,
               'buildings': 4,
               'chaparral': 5,
               'denseresidential': 6,
               'forest': 7,
               'freeway': 8,
               'golfcourse': 9,
               'harbor': 10,
               'intersection': 11,
               'mediumresidential': 12,
               'mobilehomepark': 13,
               'overpass': 14,
               'parkinglot': 15,
               'river': 16,
               'runway': 17,
               'sparseresidential': 18,
               'storagetanks': 19,
               'tenniscourt': 20,
               }

img_Width = 256
img_Height = 256
test_batch_size = 20
n_class = 21
checkpoint_path = r'/media/jsl/ubuntu/log/VGG16_UCM/logs/train'



def get_class_name_by_index(index):
    for item in class2label:
        if class2label[item] == index:
            return item

def delete_add_img(jpg_img_path):
    for type in os.listdir(jpg_img_path):
        type_path = jpg_img_path + os.sep + type
        for class_floder in os.listdir(type_path):
            class_floder_path = type_path + os.sep + class_floder
            print(class_floder_path)
            for img in os.listdir(class_floder_path):
                img_path = class_floder_path + os.sep + img
                print(img_path)
                if img.find('_') != -1:
                    os.remove(img_path)
                    print(img)


def convert_format_jpg(jpg_img_path):
    new_jpg_rootpath = jpg_img_path.replace('bak', 'new')
    print(new_jpg_rootpath)
    if not os.path.exists(new_jpg_rootpath):
        os.mkdir(new_jpg_rootpath)
    for type in os.listdir(jpg_img_path):
        type_path = jpg_img_path + os.sep + type
        print(type_path)
        new_type_path = type_path.replace('bak', 'new')
        print(new_type_path)
        if not os.path.exists(new_type_path):
            os.mkdir(new_type_path)
        for class_floder in os.listdir(type_path):
            class_floder_path = type_path + os.sep + class_floder
            new_class_folder_path = class_floder_path.replace('bak', 'new')
            if not os.path.exists(new_class_folder_path):
                os.mkdir(new_class_folder_path)
            for img in os.listdir(class_floder_path):
                img_path = class_floder_path + os.sep + img
                new_img_path = img_path.replace('bak', 'new')
                # img_content = cv2.imread(img_path)
                # cv2.imwrite(new_img_path, img_content)


if __name__ == '__main__':
    # convert_tif2jpg(tif_folder_path, jpg_folder_path)
    # split_train_test()
    # jpg_path = r'/media/jsl/ubuntu/UCMerced_LandUse/jpgImages_bak'
    # delete_add_img(jpg_path)
    # convert_format_jpg(jpg_path)
    a = get_class_name_by_index(1)
    print(a)