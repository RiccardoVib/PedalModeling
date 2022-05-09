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
    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas48_train_2.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data48_train_2.pickle'])), 'rb')

    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    Z = [inp, tar]
    Z = np.array(Z)

    M = pickle.load(meta)
    ratios = M['ratio']
    threshold = M['threshold']
    fs = M['samplerate']

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler()
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)

    ratios = np.array(ratios, dtype=np.float32)
    thresholds = np.array(threshold, dtype=np.float32)

    scaler_ratios = my_scaler()
    scaler_threshold = my_scaler()

    scaler_ratios.fit(ratios)
    scaler_threshold.fit(thresholds)
    thresholds = scaler_threshold.transform(thresholds)
    ratios = scaler_ratios.transform(ratios)

    zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)
    zero_value_ratio = (0 - scaler_ratios.min_data) / (scaler_ratios.max_data - scaler_ratios.min_data)
    zero_value_threshold = (0 - scaler_threshold.min_data) / (scaler_threshold.max_data - scaler_threshold.min_data)
    zero_value = [zero_value, zero_value_ratio, zero_value_threshold]
    scaler = [scaler, scaler_ratios, scaler_threshold]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = int(fs * w_length)
    all_inp, all_tar = [], []

    for i in range(n_record):
        for t in range(inp.shape[1] // window):
            inp_temp = np.array([inp[i, t * window:t * window + window], np.repeat(ratios[i], window),
                                 np.repeat(thresholds[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[i, t * window:t * window + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    #data = {'all_inp' : all_inp, 'all_tar': all_tar, 'scaler': scaler, 'zero_value': zero_value, 'fs': fs}
    #file_data = open(os.path.normpath('/'.join([data_dir, 'all_w1.pickle'])), 'wb')
    #pickle.dump(data, file_data)

    w = 2  # n of column
    h = len(all_inp)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        matrix[i][0] = all_inp[i]
        matrix[i][1] = all_tar[i]
    if shuffle:
        np.random.shuffle(matrix)

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
    w1 = 0.000021
    w2 = 0.000045
    w4 = 0.00009
    w8 = 0.00017
    w16 = 0.00034
    x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=27, shuffle=False, w_length=w4, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'zero_value': zero_value, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w4_limited.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()