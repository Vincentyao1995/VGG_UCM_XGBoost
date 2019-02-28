# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import cv2
import os


def ss_sub_img(img_path):
    # loading astronaut image
    # img = skimage.data.astronaut()
    # hd_parcel_ss_sub_path = r'/media/jsl/ubuntu/data/hd_parcel/hd_parcel_ss_sub'
    # img_path = r'/media/jsl/ubuntu/data/city_data/haidian_data/haidian_tif/hd_clip_jpg/hd_0295.jpg'
    # img_name = img_path.split('/')[-1].split('.')[0]
    # sub_set_img_path = os.path.join(hd_parcel_ss_sub_path, img_name)
    # if not os.path.exists(sub_set_img_path):
    #     os.mkdir(sub_set_img_path)
    img = cv2.imread(img_path)
    # print(img.shape)
    plt.imshow(img)
    # plt.show()
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.5, min_size=250)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 1800:
            continue

        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.7 or h / w > 1.7:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 0

    for x, y, w, h in candidates:
        i += 1
        # print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        sub_img = img[x:x + w, y:y + h, :]
        # sub_path = os.path.join(sub_set_img_path, 'sub_' + str(i) + '.jpg')
        # print(sub_path)
        # cv2.imwrite(sub_path, sub_img)
    plt.show()


if __name__ == "__main__":
    # hd_jpg_folder_path = r'/media/jsl/ubuntu/data/city_data/haidian_data/haidian_tif/hd_clip_jpg'
    # for img in os.listdir(hd_jpg_folder_path):
    #     print(img, "start")
    #     img_path = os.path.join(hd_jpg_folder_path, img)
    #     ss_sub_img(img_path)
    #     print(img, "end")
    img_path = r'hd_0241.jpg'
    ss_sub_img(img_path)