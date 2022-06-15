import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler
from librosa import display
import matplotlib.pyplot as plt


def get_data(data_dir, w_length, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_OD300.pickle'])), 'rb')

    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    tone = Z['tone']
    drive = Z['drive']
    mode = Z['mode']
    fs = Z['samplerate']

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    Z = [inp, tar]
    Z = np.array(Z)

    tone = np.array(tone, dtype=np.float32)
    drive = np.array(drive, dtype=np.float32)
    mode = np.array(mode, dtype=np.float32)

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler(feature_range=(-1, 1))
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    del Z
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    all_inp, all_tar = [], []

    for i in range(inp.shape[0]):
        for t in range(inp.shape[1] // w_length):
            inp_temp = np.array([inp[i, t * w_length: (t + 1)*w_length], np.repeat(tone[i], w_length), np.repeat(drive[i], w_length), np.repeat(mode[i], w_length)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[i, t * w_length: (t + 1)*w_length])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    N = inp.shape[0] * inp.shape[1] // 85

    x = np.array(all_inp[: N])
    y = np.array(all_tar[: N])
    x_val = np.array(all_inp[N+1:])
    y_val = np.array(all_tar[N+1:])

    # TEST
    all_inp, all_tar = [], []

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_test_OD300.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    tone = Z['tone']
    drive = Z['drive']
    mode = Z['mode']
    fs = Z['samplerate']
    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    tone = np.array(tone, dtype=np.float32)
    drive = np.array(drive, dtype=np.float32)
    mode = np.array(mode, dtype=np.float32)

    for t in range(inp.shape[1] // w_length):
        inp_temp = np.array(
            [inp[0, t * w_length: (t + 1)*w_length], np.repeat(tone[0], w_length), np.repeat(drive[0], w_length), np.repeat(mode[0], w_length)])
        all_inp.append(inp_temp.T)
        tar_temp = np.array(tar[0, t * w_length: (t + 1)*w_length])
        all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x_test = np.array(all_inp)
    y_test = np.array(all_tar)

    return x, y, x_val, y_val, x_test, y_test, scaler, fs

if __name__ == '__main__':

    data_dir = '../Files'

    x, y, x_val, y_val, x_test, y_test, scaler, fs = get_data(data_dir=data_dir, w_length=16, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16_OD300.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()