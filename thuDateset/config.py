IMG_W = 256
IMG_H = 256
N_CLASSES = 21
BATCH_SIZE = 16
learning_rate = 0.001
MAX_STEP = 15000  # it took me about one hour to complete the training.
IS_PRETRAIN = True
CAPACITY = 256
ucm_checkpoint_path = r'/media/jsl/ubuntu/log/ucm/logs/train'
nwpu_checkpoint_path = r''
# ------------ db config ------------
db_host = 'localhost'
db_port = 3306
db_user = 'root'
db_pw = '121314'
db_name = 'jsl_data'

# ------------ UCM dataset config ------------
ucm_class2label = {'agricultural': 0,
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
ucm_class = ['agricultural',
             'airplane',
             'baseballdiamond',
             'beach',
             'buildings',
             'chaparral',
             'denseresidential',
             'forest',
             'freeway',
             'golfcourse',
             'harbor',
             'intersection',
             'mediumresidential',
             'mobilehomepark',
             'overpass',
             'parkinglot',
             'river',
             'runway',
             'sparseresidential',
             'storagetanks',
             'tenniscourt',
             ]
ucm_root_path_fea = r'/media/jsl/ubuntu/data/UCMerced_LandUse/fea_root'
ucm_root_path_fea_tune = r'/media/jsl/ubuntu/data/UCMerced_LandUse/fea_tune_root'
ucm_root_path_fea_enhance_tune = r'/media/jsl/ubuntu/data/UCMerced_LandUse/enhance_fea_root'
ucm_xgboost_model_path = r'/media/jsl/ubuntu/model/model_xgboost_ucm/pima.pickle.dat'
ucm_svm_model_path = r'/media/jsl/ubuntu/model/model_svm_ucm/svm_ucm.dat'
ucm_data_path = r'/media/jsl/ubuntu/data/UCMerced_LandUse/jpgImages_rotate/jpgImages_rotate'
ucm_n_class = 21
# ------------ NWPU config------------
nwpu_class = ['airplane',
                    'airport',
                    'baseball_diamond',
                    'basketball_court',
                    'beach',
                    'bridge',
                    'chaparral',
                    'church',
                    'circular_farmland',
                    'cloud',
                    'commercial_area',
                    'dense_residential',
                    'desert',
                    'forest',
                    'freeway',
                    'golf_course',
                    'ground_track_field',
                    'harbor',
                    'industrial_area',
                    'intersection',
                    'island',
                    'lake',
                    'meadow',
                    'medium_residential',
                    'mobile_home_park',
                    'mountain',
                    'overpass',
                    'palace',
                    'parking_lot',
                    'railway',
                    'railway_station',
                    'rectangular_farmland',
                    'river',
                    'roundabout',
                    'runway',
                    'sea_ice',
                    'ship',
                    'snowberg',
                    'sparse_residential',
                    'stadium',
                    'storage_tank',
                    'tennis_court',
                    'terrace',
                    'thermal_power_station',
                    'wetland']

nwpu_class2label = {'airplane': 0,
                    'airport': 1,
                    'baseball_diamond': 2,
                    'basketball_court': 3,
                    'beach': 4,
                    'bridge': 5,
                    'chaparral': 6,
                    'church': 7,
                    'circular_farmland': 8,
                    'cloud': 9,
                    'commercial_area': 10,
                    'dense_residential': 11,
                    'desert': 12,
                    'forest': 13,
                    'freeway': 14,
                    'golf_course': 15,
                    'ground_track_field': 16,
                    'harbor': 17,
                    'industrial_area': 18,
                    'intersection': 19,
                    'island': 20,
                    'lake': 21,
                    'meadow': 22,
                    'medium_residential': 23,
                    'mobile_home_park': 24,
                    'mountain': 25,
                    'overpass': 26,
                    'palace': 27,
                    'parking_lot': 28,
                    'railway': 29,
                    'railway_station': 30,
                    'rectangular_farmland': 31,
                    'river': 32,
                    'roundabout': 33,
                    'runway': 34,
                    'sea_ice': 35,
                    'ship': 36,
                    'snowberg': 37,
                    'sparse_residential': 38,
                    'stadium': 39,
                    'storage_tank': 40,
                    'tennis_court': 41,
                    'terrace': 42,
                    'thermal_power_station': 43,
                    'wetland': 44}

nwpu_root_path_fea_tune = r'/media/jsl/ubuntu/data/NWPU/NWPU_fea/NWPU_fea'
nwpu_xgboost_model_path = r'/media/jsl/ubuntu/model/model_xgboost_nwpu/pima.pickle.dat'
nwpu_data_path = r'/media/jsl/ubuntu/data/NWPU-RESISC45/'
nwpu_n_class = 45

# ------------ AID config ------------

aid_data_root_path = r'/media/jsl/ubuntu/data/AID/AID_dataset/AID'
aid_fea_root_path = r''
aid_checkpoint_path = r''
aid_log_root_path = r'/media/jsl/ubuntu/log/AID/logs'
aid_result_root_path = r''
aid_n_class = 30
aid_img_height = 600
aid_img_weight = 600
aid_class2label = {'Airport': 0,
                   'BareLand': 1,
                   'BaseballField': 2,
                   'Beach': 3,
                   'Bridge': 4,
                   'Center': 5,
                   'Church': 6,
                   'Commercial': 7,
                   'DenseResidential': 8,
                   'Desert': 9,
                   'Farmland': 10,
                   'Forest': 11,
                   'Industrial': 12,
                   'Meadow': 13,
                   'MediumResidential': 14,
                   'Mountain': 15,
                   'Park': 16,
                   'Parking': 17,
                   'Playground': 18,
                   'Pond': 19,
                   'Port': 20,
                   'RailwayStation': 21,
                   'Resort': 22,
                   'River': 23,
                   'School': 24,
                   'SparseResidential': 25,
                   'Square': 26,
                   'Stadium': 27,
                   'StorageTanks': 28,
                   'Viaduct': 29}

# ----------- hd_clip_parcel config ------------

hd_clip_parcel_class2label = {
    'RES': 0,
    'EDU': 1,
    'TRA': 2,
    'GRE': 3,
    'COM': 4,
    'OTH': 5
}
hd_clip_parcel_label_path = r'/media/jsl/ubuntu/data/hd_parcel_spp_fea/hd_clip_parcel_type.txt'
hd_clip_parcel_folder_path = r'/media/jsl/ubuntu/data/hd_parcel_spp_fea/'
