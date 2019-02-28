import sys
import config
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

sys.path.append('./')
from classifier_xgboost import get_enhance_fea_vec, get_1class_fea
from sklearn.metrics import accuracy_score
import time
import os


def fit_svm_ucm():
    train, label_train, val, label_val, test, label_test = get_enhance_fea_vec(config.ucm_root_path_fea_enhance_tune)
    print(len(train), len(label_train))
    print(len(val), len(label_val))
    print(len(test), len(label_test))
    train = np.array(train)
    label_train = np.array(label_train)
    test = np.array(test)
    label_test = np.array(label_test)
    watchlist = [(train, 'train'), (test, 'val')]
    # Train the Linear SVM
    clf = LinearSVC()
    start_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    print('end_time:', start_time)
    clf.fit(train, label_train)
    end_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    print('end_time:', end_time)
    pickle.dump(clf, open('svm_ucm.dat', 'wb'))

    y_pred = clf.predict(test)
    print(y_pred)
    accuracy = accuracy_score(label_test, y_pred)
    print(accuracy)


def test_svm_ucm():
    model = pickle.load(open(config.ucm_svm_model_path, 'rb'))
    matrix_confusion = np.zeros((config.ucm_n_class, config.ucm_n_class))

    # print(matrix_confusion)
    for class_name in config.ucm_class2label:
        print(class_name)
        test, label_test = get_1class_fea(os.path.join(config.ucm_root_path_fea_enhance_tune, 'validation'), class_name)
        print(len(test), len(label_test))
        test = np.array(test)
        label_test = np.array(label_test)
        y_pred = model.predict(test)
        accuracy = accuracy_score(label_test, y_pred)
        print(accuracy)
        class_index = config.ucm_class2label[class_name]
        for item in y_pred:
            matrix_confusion[class_index][item] += 1
    np.savetxt('ucm_svm_confusion_matrix', matrix_confusion)


if __name__ == "__main__":
    fit_svm_ucm()
    # test_svm_ucm()