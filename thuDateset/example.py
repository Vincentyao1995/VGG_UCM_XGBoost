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

def main():

    # loading astronaut image
    # img = skimage.data.astronaut()
    img_path = 'hd_0613.jpg'
    img = cv2.imread(img_path)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.8, min_size=20)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 0

    for x, y, w, h in candidates:
        i += 1
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        sub_img = img[x:x+w, y:y+h, :]
        sub_path = 'sub_'+str(i)+'.jpg'
        cv2.imwrite(sub_path, sub_img)
    plt.show()

if __name__ == "__main__":
    main()