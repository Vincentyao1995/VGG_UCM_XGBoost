import cv2
import gdal
import matplotlib.pyplot as plt
from libtiff import TIFF
import numpy as np


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





if __name__ == '__main__':
    tif_path = r'/media/jsl/ubuntu/test/hd_001.tif'
    jpg_path = r'/media/jsl/ubuntu/test/33.jpg'
    convert_img_tif2jpg(tif_path, jpg_path)
