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
    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas_test.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_test.pickle'])), 'rb')

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

    x_test, y_test = [], []

    window = w_length
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

    return x_test, y_test, scaler, zero_value, fs

if __name__ == '__main__':

    data_dir = '../Files'
    w1 = 1
    w2 = 2
    w4 = 4
    w8 = 8
    w16 = 16
    x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=27, shuffle=False, w_length=w16, seed=422)

    data = {'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'zero_value': zero_value, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16_test_samples.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()