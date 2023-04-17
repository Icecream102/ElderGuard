import pickle
from sklearn.model_selection import train_test_split
import numpy as np
# 解决colab引用路径问题
import sys
sys.path.append('/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master')
from Actionsrecognition.Models import *


# 划分数据,全部加载进来打乱后按6：2：2划分
def load_dataset():
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    data_files = ['/content/drive/MyDrive/ElderGuard/data/input_data/train_data.pkl',
                  '/content/drive/MyDrive/ElderGuard/data/input_data/val_data.pkl',
                  '/content/drive/MyDrive/ElderGuard/data/input_data/test_data.pkl']
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
            # print('fts shape:', fts.shape)
            # print('lbs shape:', lbs.shape)

        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    x_train, x_left, y_train, y_left = train_test_split(features, labels, test_size=0.4,
                                                        random_state=9)
    x_valid, x_test, y_valid, y_test = train_test_split(x_left, y_left, test_size=0.5,
                                                        random_state=9)
    print('train_shape:', x_train.shape)
    print('valid_shape:', x_valid.shape)
    print('test_shape:', x_test.shape)
    train_p = '/content/drive/MyDrive/ElderGuard/data/input_data/train_data_2.pkl'  # 保存训练集,用于test_model.py
    with open(train_p, 'wb') as f:
        pickle.dump((x_train, y_train), f)

    valid_p = '/content/drive/MyDrive/ElderGuard/data/input_data/valid_data_2.pkl'  # 保存训练集,用于test_model.py
    with open(valid_p, 'wb') as f:
        pickle.dump((x_valid, y_valid), f)

    test_p = '/content/drive/MyDrive/ElderGuard/data/input_data/test_data_2.pkl'  # 保存训练集,用于test_model.py
    with open(test_p, 'wb') as f:
        pickle.dump((x_test, y_test), f)

load_dataset()
