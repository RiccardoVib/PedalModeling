import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler


def get_data(data_dir, w_length, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    meta = open(os.path.normpath('/'.join([data_dir, 'metadatasOD300_cond1.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'dataOD300_cond1.pickle'])), 'rb')

    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    Z = [inp, tar]
    Z = np.array(Z)

    M = pickle.load(meta)
    #tone = M['tone']
    #drive = M['drive']
    #mode = M['mode']
    fs = M['samplerate']
    #tone = np.array(tone, dtype=np.float32)
    #drive = np.array(drive, dtype=np.float32)
    #mode = np.array(mode, dtype=np.float32)

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler()
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    del Z
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = w_length
    all_inp, all_tar = [], []

    for i in range(inp.shape[0]):
        for t in range(inp.shape[1] // window):
            inp_temp = np.array([inp[i, t * window:t * window + window]])#, np.repeat(tone[i], window)])
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
    n_train = N // 100 * 85
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

    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas_test_OD300_cond1.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_test_OD300_cond1.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    #M = pickle.load(meta)
    #tone = M['tone'] / 2
    #drive = M['drive'] / 2
    #mode = M['mode'] / 2
    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    #tone = np.array(tone, dtype=np.float32)
    #drive = np.array(drive, dtype=np.float32)
    #mode = np.array(mode, dtype=np.float32)

    for t in range(inp.shape[1] // window):
        inp_temp = np.array(
            [inp[0, t * window:t * window + window]])#, np.repeat(tone[0], window)])
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

    return x, y, x_val, y_val, x_test, y_test, scaler, fs

if __name__ == '__main__':

    data_dir = '../Files'

    x, y, x_val, y_val, x_test, y_test, scaler, fs = get_data(data_dir=data_dir, w_length=16, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16_OD300_nocond.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()