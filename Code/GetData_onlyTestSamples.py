import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Preprocess import my_scaler
from scipy.io import wavfile

def get_data(data_dir, sets, T, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16.pickle'])), 'rb')
    Z = pickle.load(file_data)
    scaler = Z['scaler']

    meta = open(os.path.normpath('/'.join([data_dir, 'meta_samples_collection.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'test_samples_collection.pickle'])), 'rb')
    #file_scaler = open(os.path.normpath('/'.join([data_dir, 'scaler.pickle'])), 'rb')
    #scaler = pickle.load(file_scaler)
    #scaler = scaler['scaler']
    Z = pickle.load(file_data)
    M = pickle.load(meta)

    if sets != 'test':
        Z = Z[sets]
        sweep_inp = Z['sweep_inp']
        sweep_tar = Z['sweep_tar']

        #tar_name = '_sweep_' + str(sets) + '_tar.wav'
        #tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #tar = sweep_tar.astype('int16')
        #wavfile.write(tar_dir, 48000, tar)

        guitar_inp = Z['guitar_inp']
        guitar_tar = Z['guitar_tar']

        #tar_name = '_guitar_' + str(sets) + '_tar.wav'
        #tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #tar = guitar_tar.astype('int16')
        #wavfile.write(tar_dir, 48000, tar)

        kick_inp = Z['drumKick_inp']
        kick_tar = Z['drumKick_tar']

        #tar_name = '_drumKick_' + str(sets) + '_tar.wav'
        #tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #tar = kick_tar.astype('int16')
        #wavfile.write(tar_dir, 48000, tar)

        hh_inp = Z['drumHH_inp']
        hh_tar = Z['drumHH_tar']

        #tar_name = '_drumHH_' + str(sets) + '_tar.wav'
        #tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #tar = hh_tar.astype('int16')
        #wavfile.write(tar_dir, 48000, tar)

        bass_inp = Z['bass_inp']
        bass_tar = Z['bass_tar']

        #tar_name = '_bass_' + str(sets) + '_tar.wav'
        #tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #tar = bass_tar.astype('int16')
        #wavfile.write(tar_dir, 48000, tar)

        M = M[sets]
        ratio = M['ratio']
        threshold = M['threshold']
    else:
        Z = Z[16]
        sweep_inp = Z['sweep_inp_test']
        sweep_tar = Z['sweep_tar_test']

        guitar_inp = Z['guitar_inp_test']
        guitar_tar = Z['guitar_tar_test']

        kick_inp = Z['drumKick_inp_test']
        kick_tar = Z['drumKick_tar_test']

        hh_inp = Z['drumHH_inp_test']
        hh_tar = Z['drumHH_tar_test']

        bass_inp = Z['bass_inp_test']
        bass_tar = Z['bass_tar_test']

        M = M[16]
        ratio = M['ratio_test']
        threshold = M['threshold_test']



    sweep_inp = scaler[0].transform(np.array(sweep_inp, dtype=np.float32))
    sweep_tar = scaler[0].transform(np.array(sweep_tar, dtype=np.float32))
    guitar_inp = scaler[0].transform(np.array(guitar_inp, dtype=np.float32))
    guitar_tar = scaler[0].transform(np.array(guitar_tar, dtype=np.float32))
    kick_inp = scaler[0].transform(np.array(kick_inp, dtype=np.float32))
    kick_tar = scaler[0].transform(np.array(kick_tar, dtype=np.float32))
    hh_inp = scaler[0].transform(np.array(hh_inp, dtype=np.float32))
    hh_tar = scaler[0].transform(np.array(hh_tar, dtype=np.float32))
    bass_inp = scaler[0].transform(np.array(bass_inp, dtype=np.float32))
    bass_tar = scaler[0].transform(np.array(bass_tar, dtype=np.float32))
    inp = [sweep_inp, guitar_inp, kick_inp, hh_inp, bass_inp]
    tar = [sweep_tar, guitar_tar, kick_tar, hh_tar, bass_tar]

    ratio = scaler[1].transform(np.array(ratio, dtype=np.float32))
    threshold = scaler[2].transform(np.array(threshold, dtype=np.float32))
    window = T
    signals_inp, signals_tar = [], []

    for i in range(len(inp)):
        all_inp, all_tar = [], []
        x_, y_ = [], []
        inp_ = inp[i]
        tar_ = tar[i]
        for t in range(inp_.shape[0] - window):
            inp_temp = np.array([inp_[t:t + window], np.repeat(ratio, window), np.repeat(threshold, window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar_[t:t + window])
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

        signals_inp.append(np.array(x_))
        signals_tar.append(np.array(y_))

    return signals_inp, signals_tar, scaler

if __name__ == '__main__':

    data_dir = '../Files'

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16.pickle'])), 'rb')
    data = pickle.load(file_data)
    x_test = data['x_test']
    T = x_test.shape[1]

    scaler = data['scaler']

    signals_inp, signals_tar , scaler = get_data(data_dir=data_dir, sets=3, scaler=scaler, T=T, seed=422)