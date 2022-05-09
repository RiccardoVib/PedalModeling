import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Preprocess import my_scaler


def get_data(data_dir, start, stop, T, seed=422):
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

    ratios = np.array(ratios, dtype=np.float32)
    thresholds = np.array(threshold, dtype=np.float32)

    scaler_ratios = my_scaler()
    scaler_threshold = my_scaler()
    scaler_ratios.fit(ratios)
    scaler_threshold.fit(thresholds)

    scaler = [scaler, scaler_ratios, scaler_threshold]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x_, y_ = [], []

    # TEST
    all_inp, all_tar = [], []

    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas48_test_2.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data48_test_2.pickle'])), 'rb')
    #meta = open(os.path.normpath('/'.join([data_dir, 'metadatas466_-10.pickle'])), 'rb')
    #file_data = open(os.path.normpath('/'.join([data_dir, 'data466_-10.pickle'])), 'rb')
    #meta = open(os.path.normpath('/'.join([data_dir, 'metadatas733_-40.pickle'])), 'rb')
    #file_data = open(os.path.normpath('/'.join([data_dir, 'data733_-40.pickle'])), 'rb')

    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    inp = np.array(inp, dtype=np.float32)
    inp = inp[:, start:stop]
    tar = np.array(tar, dtype=np.float32)
    tar = tar[:, start:stop]
    M = pickle.load(meta)
    ratios = M['ratio']
    threshold = M['threshold']
    inp = scaler[0].transform(inp)
    tar = scaler[0].transform(tar)
    ratios = np.array(ratios, dtype=np.float32)
    thresholds = np.array(threshold, dtype=np.float32)
    thresholds = scaler[2].transform(thresholds)
    ratios = scaler[1].transform(ratios)
    window = T
    for t in range(inp.shape[1] - window):
        inp_temp = np.array(
            [inp[0, t:t + window], np.repeat(ratios[0], window), np.repeat(thresholds[0], window)])
        all_inp.append(inp_temp.T)
        tar_temp = np.array(tar[0, t:t + window])
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
        x_.append(matrix[n][0])
        y_.append(matrix[n][1])

    x_ = np.array(x_)
    y_ = np.array(y_)

    return x_, y_, scaler

if __name__ == '__main__':

    data_dir = '../Files'

    x_, y_ , scaler = get_data(data_dir=data_dir, seed=422)

    data = {'x': x_, 'y': y_}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w2_for_v2.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()