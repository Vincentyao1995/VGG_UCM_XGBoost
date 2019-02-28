import numpy as np
import config
import os
import scipy
import pandas
import xgboost as xgb
import config
import os
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV


def get_data():
    label = []
    with open(config.hd_clip_parcel_label_path) as f:
        line = f.readline()
        line = f.readline().strip()
        num = 0
        while line:
            num += 1
            type = config.hd_clip_parcel_class2label[line.split(',')[-1]]
            label.append(float(type))
            line = f.readline().strip()
    label = np.array(label)
    print(label)
    data = []
    for parcel_name in os.listdir(config.hd_clip_parcel_folder_path):
        suffix = parcel_name.split('.')[-1]
        if suffix == 'npy':
            num = int(parcel_name.split('.')[0].split('_')[-1])

            fea = np.load(os.path.join(config.hd_clip_parcel_folder_path, parcel_name))
            fea_reshape = np.array([])
            for f in fea:
                f2 = f.reshape(-1)
                fea_reshape = np.append(fea_reshape, f2)
            data.append(fea_reshape)
    data = np.array(data)
    data = np.transpose(data)
    print(data.shape)
    data_label = np.vstack((data, label))
    print(data_label.shape)
    data_label = np.transpose(data_label)
    print(data_label.shape)
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    num6 = 0
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for fea in data_label:
        # print(fea[-1])
        # print(fea[0:15361])
        type = fea[-1]
        if type == 0.0:
            num1 += 1
            if num1 <= 320:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
        elif type == 1.0:
            num2 += 1
            if num2 <= 160:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
        elif type == 2.0:
            num3 += 1
            if num3 <= 21:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
        elif type == 3.0:
            num4 += 1
            if num4 <= 240:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
        elif type == 4.0:
            num5 += 1
            if num5 <= 67:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
        elif type == 5.0:
            num6 += 1
            if num6 <= 103:
                train_data.append(fea[0:15361])
                train_label.append(fea[-1])
            else:
                test_data.append(fea[0:15361])
                test_label.append(fea[-1])
    return train_data, train_label, test_data, test_label


def fit_xgboost_hd_clip_parcel():
    train, label_train, test, label_test = get_data()
    train = np.array(train)
    label_train = np.array(label_train)
    test = np.array(test)
    label_test = np.array(label_test)
    watchlist = [(train, 'train'), (test, 'val')]
    model = XGBClassifier()
    model.fit(train, label_train)
    y_pred = model.predict(test)
    print(y_pred)
    accuracy = accuracy_score(label_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    fit_xgboost_hd_clip_parcel()
