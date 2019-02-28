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
import pickle
import matplotlib.pyplot as plt


def get_fea_vec(fea_root_path):
    train = []
    label_train = []
    test = []
    label_test = []
    for class_name in os.listdir(fea_root_path):
        class_path = os.path.join(fea_root_path, class_name)
        class_label = config.ucm_class2label[class_name]
        print(class_name, class_label)
        index = 0
        for fea_txt_name in os.listdir(class_path):
            fea_txt_path = os.path.join(class_path, fea_txt_name)
            with open(fea_txt_path, 'r') as f:
                fea_str = f.read()
                fea_vec = [float(i) for i in fea_str.split(',')]
                if index % 5:
                    train.append(fea_vec)
                    label_train.append(class_label)
                else:
                    test.append(fea_vec)
                    label_test.append(class_label)
                index += 1

    return train, label_train, test, label_test


def get_enhance_fea_vec(enhance_fea_root_path):
    train = []
    label_train = []
    val = []
    label_val = []
    test = []
    label_test = []
    for data_type in os.listdir(enhance_fea_root_path):
        data_type_path = os.path.join(enhance_fea_root_path, data_type)
        for class_name in os.listdir(data_type_path):
            class_path = os.path.join(data_type_path, class_name)
            class_label = config.ucm_class2label[class_name]
            print(class_name, class_label)

            for fea_txt_name in os.listdir(class_path):
                fea_txt_path = os.path.join(class_path, fea_txt_name)
                with open(fea_txt_path, 'r') as f:
                    fea_str = f.read()
                    fea_vec = [float(i) for i in fea_str.split(',')]
                    if data_type == 'train':
                        train.append(fea_vec)
                        label_train.append(class_label)
                    elif data_type == 'validation':
                        val.append(fea_vec)
                        label_val.append(class_label)
                    else:
                        test.append(fea_vec)
                        label_test.append(class_label)
    return train, label_train, val, label_val, test, label_test


def get_1class_fea(test_fea_root_path, tar_class_name):
    test = []
    label_test = []
    for class_name in os.listdir(test_fea_root_path):
        if class_name == tar_class_name:
            class_label = config.ucm_class2label[class_name]
            class_path = os.path.join(test_fea_root_path, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                with open(file_path, 'r') as f:
                    fea_str = f.read()
                    fea_vec = [float(i) for i in fea_str.split(',')]
                    # print(len(fea_vec))
                    test.append(fea_vec)
                    label_test.append(class_label)
    return test, label_test


def test_xgboost_ucm():
    # test, label_test = get_1class_fea(os.path.join(config.ucm_root_path_fea_enhance_tune, 'validation'), 'beach')
    # test = np.array(test)
    # label_test = np.array(label_test)
    # print(len(test))
    # print(label_test)
    # model = pickle.load(open(config.ucm_xgboost_model_path, 'rb'))
    # y_pred = model.predict(test)
    # print(y_pred)
    model = pickle.load(open(config.nwpu_xgboost_model_path, 'rb'))
    matrix_confusion = np.zeros((config.nwpu_n_class, config.nwpu_n_class))

    # print(matrix_confusion)
    for class_name in config.ucm_class2label:
        print(class_name)
        test, label_test = get_1class_fea(os.path.join(config.nwpu_root_path_fea_tune, 'val'), class_name)
        print(len(test), len(label_test))
        test = np.array(test)
        label_test = np.array(label_test)
        y_pred = model.predict(test)
        accuracy = accuracy_score(label_test, y_pred)
        print(accuracy)
        class_index = config.ucm_class2label[class_name]
        for item in y_pred:
            matrix_confusion[class_index][item] += 1
    np.savetxt('nwpu_xgboost_confusion_matrix', matrix_confusion)


def fit_xgboost_ucm():
    train, label_train, val, label_val, test, label_test = get_enhance_fea_vec(config.ucm_root_path_fea_enhance_tune)
    print(len(train), len(label_train))
    print(len(val), len(label_val))
    print(len(test), len(label_test))
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


def draw_matrix_confusion(matrix_confusion_path):
    matrix = np.loadtxt(matrix_confusion_path)
    norm_conf = []
    over_acc = 0
    x = 0
    for i in matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        y = 0
        for j in i:
            tmp_arr.append(float(j) / float(a))
            if x == y:
                over_acc += float(j) / float(a)
                print(float(j) / float(a))
            y += 1

        norm_conf.append(tmp_arr)
        x += 1
    print(over_acc / 45)

    fig = plt.figure(num=10, figsize=(16, 9), dpi=200)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(4)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = matrix.shape

    # for x in range(width):
    #     for y in range(height):
    #         if not norm_conf[x][y] == 0:
    #             ax.annotate(str(norm_conf[x][y]), xy=(y, x), size=7, color='w',
    #             horizontalalignment='center',
    #             verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), config.nwpu_class[:width], rotation=90)
    keys = config.nwpu_class2label.keys()

    plt.yticks(range(height), config.nwpu_class[:height])
    plt.savefig('svm_nwpu_confusion_matrix.png', format='png')


def getKappa_and_each_class_accuracy():
    confusion_matirx_file = r'/media/jsl/ubuntu/result/nwpu/svm/nwpu_svm_confusion_matrix'
    confusion_matrix = np.loadtxt(confusion_matirx_file)
    print(confusion_matrix.shape)
    print(confusion_matrix[3])
    accuracy_one_class = 0
    accuracy_all = 0
    for i in range(0, confusion_matrix.shape[0]):
        accuracy_one_class = confusion_matrix[i][i]/80
        accuracy_all += accuracy_one_class
        print(i, ":", accuracy_one_class)

    accuracy_all = accuracy_all / 45
    print("OA:", accuracy_all)
    raw_sum = confusion_matrix.sum(axis=1)
    col_sum = confusion_matrix.sum(axis=0)
    p0 = accuracy_all
    pe = np.sum(raw_sum * col_sum) / (raw_sum.sum() * col_sum.sum())

    kappa = (p0 - pe) / (1 - pe)
    print("kappa:", kappa)


# # read in data
# dtrain = xgb.DMatrix('./data/agaricus.txt.train')
# print(dtrain[0])
# dtest = xgb.DMatrix('./data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)
# print(preds)

def generate_nwpu_vgg_confusion_matrix():
    nwpu_vgg_confusion_matrx_num_path = r"/media/jsl/ubuntu/result/nwpu/vgg/confuse_matrix_update_num.txt"
    nwpu_vgg_confusion_matrx_num_percent = r"/media/jsl/ubuntu/result/nwpu/vgg/confuse_matrix_update_percent.txt"
    confusion_matrix = np.zeros((45, 45))
    with open(nwpu_vgg_confusion_matrx_num_path, 'r') as f:
        line = f.readline()
        i = 0
        while line is not "":
            line = line.strip().split(',')
            for j in range(0, 45):
                print(i, ",", j, int(j) / 120)
                confusion_matrix[i][j] = int(line[j]) / 120
            line = f.readline()
            i += 1
    np.savetxt(nwpu_vgg_confusion_matrx_num_percent, confusion_matrix)


if __name__ == '__main__':
    # test_xgboost_ucm()
    # ucm_matrix_confusion_path = r'/media/jsl/ubuntu/result/ucm/svm/ucm_svm_confusion_matrix'
    # draw_matrix_confusion(ucm_matrix_confusion_path)
    # getKappa_and_each_class_accuracy()
    # nwpu_xgboost_matrix_confusion_path = r"/media/jsl/ubuntu/result/nwpu/xgboost/nwpu_xgboost_confusion_matrix"
    # draw_matrix_confusion(nwpu_xgboost_matrix_confusion_path)
    # generate_nwpu_vgg_confusion_matrix()
    # nwpu_vgg_confusion_matrx_num_percent = r"/media/jsl/ubuntu/result/nwpu/vgg/confuse_matrix_update_percent.txt"
    # confusion_matirx_file = r'/media/jsl/ubuntu/result/nwpu/svm/nwpu_svm_confusion_matrix'
    # draw_matrix_confusion(confusion_matirx_file)
    getKappa_and_each_class_accuracy()