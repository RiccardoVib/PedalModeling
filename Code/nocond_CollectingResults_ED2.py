from ProperEvaluationAllModels import load_audio, prediction_accuracy, measure_performance, measure_time, load_model_dense, load_model_lstm
from ProperEvaluationAllModels import load_model_lstm_enc_dec, load_model_lstm_enc_dec_v2, inferenceLSTM_enc_dec, inferenceLSTM_enc_dec_v2
from ProperEvaluationAllModels import plot_time, plot_fft, create_ref, spectrogram, load_ref
import os
import pickle
import glob
import audio_format
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import fft
from mag_smoothing import mag_smoothing
from sklearn import metrics

def retrive_info(architecture, model_dir, units, drop, w):

    data_ = '../Files'
    fs = 48000
    dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/'
    data_dir = os.path.normpath(os.path.join(dir, model_dir))

    name = 'LSTM_enc_dec_v2'
    T = 16
    enc_units = [units[0]]
    dec_units = [units[1]]
    data_dir_never = '../Files'
    file_never = open(os.path.normpath('/'.join([data_dir_never, 'data48_never_seen.pickle'])), 'rb')
    file_meta_never = open(os.path.normpath('/'.join([data_dir_never, 'metadatas48_never_seen.pickle'])), 'rb')
    data_never = pickle.load(file_never)
    meta_never = pickle.load(file_meta_never)




    # LSTM_enc_dec_v2-------------------------------------------------------------------------
    if architecture == 'lstm_enc_dec_v2':
        data_dir_ref = '/Users/riccardosimionato/PycharmProjects/All_Results'
        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]

        inp, tar, fs = load_ref()
        all_results = []
        for l in range(len(sig_name)):
            start = int(sec[l] * fs)
            stop = int(sec_end[l] * start)
            file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + '_pred.wav'])))

            for file in file_pred:
                _, audio_pred = wavfile.read(file)

            audio_inp = inp[start:stop].astype('int16')
            audio_tar = tar[start:stop].astype('int16')
            audio_inp = audio_inp[w:]
            audio_tar = audio_tar[w:]
            audio_inp = audio_format.pcm2float(audio_inp)
            audio_tar = audio_format.pcm2float(audio_tar)
            audio_pred = audio_format.pcm2float(audio_pred)
            #plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
            #plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
            #results = measure_performance(audio_tar, audio_pred, name)
            #all_results.append(results)

            #spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, sig_name[l] + name)
            #print(sig_name[l], ' : ', metrics.mean_squared_error(audio_tar, audio_pred))
        # with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
        #     i=0
        #     for res in all_results:
        #         print('\n', 'Sound', '  : ', sig_name[i], file=f)
        #         i=i+1
        #         for key, value in res.items():
        #             print('\n', key, '  : ', value, file=f)



if __name__ == '__main__':
#lstm_enc_dec_v2
    retrive_info(architecture=' ', model_dir='LSTM_enc_dec_v2_16_64_64', units=[64, 64], drop=0., w=16)
