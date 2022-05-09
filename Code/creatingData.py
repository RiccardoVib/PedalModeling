import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Preprocess import my_scaler


def get_data(data_dir, n_record, shuffle, w_length, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
#260000000
    data_1 = open(os.path.normpath('/'.join([data_dir, 'data_OD300_1.pickle'])), 'rb')
    data_2 = open(os.path.normpath('/'.join([data_dir, 'data_OD300_2.pickle'])), 'rb')

    Z1 = pickle.load(data_1)
    Z2 = pickle.load(data_2)

    all_inp_1 = Z1['all_inp']
    all_tar_1 = Z1['all_tar']

    all_inp_2 = Z2['all_inp']
    all_tar_2 = Z2['all_tar']

    all_inp = np.concatenate((all_inp_1, all_inp_2))
    all_tar = np.concatenate((all_tar_1, all_tar_2))

    del all_inp_1, all_inp_2, all_tar_1, all_tar_2, Z1, Z2

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = w_length

    w = 2  # n of column
    h = len(all_inp)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        matrix[i][0] = all_inp[i]
        matrix[i][1] = all_tar[i]

    N = all_inp.shape[0]
    n_train = N // 100 * 70
    n_val = (N - n_train)

    for n in range(n_train):
        x.append(matrix[n][0])
        y.append(matrix[n][1])

    for n in range(n_val):
        x_val.append(matrix[n_train + n][0])
        y_val.append(matrix[n_train + n][1])

    x = np.array(x)
    y = np.array(y)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # TEST
    all_inp, all_tar = [], []

    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas48_test_2.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data48_test_2.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    M = pickle.load(meta)
    ratios = M['ratio']
    threshold = M['threshold']
    inp = scaler[0].transform(inp)
    tar = scaler[0].transform(tar)
    ratios = np.array(ratios, dtype=np.float32)
    thresholds = np.array(threshold, dtype=np.float32)
    thresholds = scaler[2].transform(thresholds)
    ratios = scaler[1].transform(ratios)

    for t in range(inp.shape[1] // window):
        inp_temp = np.array(
            [inp[0, t * window:t * window + window], np.repeat(ratios[0], window), np.repeat(thresholds[0], window)])
        all_inp.append(inp_temp.T)
        tar_temp = np.array(tar[0, t * window:t * window + window])
        all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    w = 2  # n of column
    h = len(all_inp)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        matrix[i][0] = all_inp[i]
        matrix[i][1] = all_tar[i]

    N = all_inp.shape[0]
    for n in range(N):
        x_test.append(matrix[n][0])
        y_test.append(matrix[n][1])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs

if __name__ == '__main__':

    data_dir = '../Files'
    w1 = 1
    w2 = 2
    w4 = 4
    w8 = 8
    w16 = 16
    x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=27, shuffle=False, w_length=w1, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'zero_value': zero_value, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w1_limited.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()