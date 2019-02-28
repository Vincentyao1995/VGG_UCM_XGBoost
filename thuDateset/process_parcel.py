import cv2
import os
import utils
import sys
import utils_trans_coor
import json


def convert_all_img_tif2jpg(tif_folder_path, jpg_folder_path):
    for tif_name in os.listdir(tif_folder_path):
        if tif_name[-3:] == 'tif':
            num = tif_name.split('.')[0].split('_')[1]
            tif_path = os.path.join(tif_folder_path, tif_name)
            if len(num) < 4:
                num_new = '0' + num
            else:
                num_new = num
            jpg_name = tif_name.replace('tif', 'jpg').replace(num, num_new)
            jpg_path = os.path.join(jpg_folder_path, jpg_name)
            utils.convert_img_tif2jpg(tif_path, jpg_path)

def convert_parcel_wgs84_to_gcj02(gjson_path_wgs84, gjson_path_gcj02):
    f = open(gjson_path_wgs84)
    wgs84_json = json.loads(f.read())
    features = wgs84_json['features']
    # print(features)
    for feature in features:
        coor = feature['geometry']['coordinates']
        for part in coor:
            for pt in part:
                print(pt)
                lat = pt[1]
                lng = pt[0]
                pt[1], pt[0] = utils_trans_coor.wgs2gcj(lat,lng)

    with open(gjson_path_gcj02, 'w') as f:
        f.write(json.dumps(wgs84_json, ))



if __name__ == '__main__':
    tif_folder_path = r'/media/jsl/ubuntu/data/city_data/haidian_data/haidian_tif/hd_clip'
    jpg_folder_path = r'/media/jsl/ubuntu/data/city_data/haidian_data/haidian_tif/hd_clip_jpg'
    hd_parcel_gjson_path_wgs84 = r'/media/jsl/ubuntu/data/parcel/haidian_clip_parcels_wgs84.geojson'
    hd_parcel_gjson_path_gcj02 = r'/media/jsl/ubuntu/data/parcel/haidian_clip_parcels_gcj02.geojson'

    # convert_all_img_tif2jpg(tif_folder_path, jpg_folder_path)
    # convert_parcel_wgs84_to_gcj02(hd_parcel_gjson_path_wgs84, hd_parcel_gjson_path_gcj02)