from ProperEvaluationAllModels import load_audio, prediction_accuracy, measure_performance, measure_time, load_model_dense, load_model_lstm
from ProperEvaluationAllModels import load_model_lstm_enc_dec, load_model_lstm_enc_dec_v2, inferenceLSTM_enc_dec, inferenceLSTM_enc_dec_v2
from ProperEvaluationAllModels import plot_time, plot_fft, create_ref, spectrogram
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

model_dir='LSTM_enc_dec_2'
units=[8, 8]
drop=0
w=2
data_ = '../Files'
data_dir_ref = '/Users/riccardosimionato/PycharmProjects/All_Results'
dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_trials/'
data_dir = os.path.normpath(os.path.join(dir, model_dir))
# data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_trials/LSTM_enc_dec_32_32'

file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w2.pickle'])), 'rb')
data = pickle.load(file_data)
x_test = data['x_test']
y_test = data['y_test']
fs = data['fs']
scaler = data['scaler']

name = 'LSTM_enc_dec'
T = x_test.shape[1]
enc_units = [units[0]]
dec_units = [units[1]]

sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
sec = [32, 135, 238, 240.9, 308.7]
sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]

all_results = []
for l in range(len(sig_name)):
    file_inp = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_inp.wav'])))
    file_tar = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_tar.wav'])))
    file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + '_pred.wav'])))
    for file in file_tar:
        fs, audio_tar = wavfile.read(file)
    for file in file_pred:
        _, audio_pred = wavfile.read(file)
    for file in file_inp:
        fs, audio_inp = wavfile.read(file)

    audio_inp = audio_format.pcm2float(audio_inp)
    audio_tar = audio_format.pcm2float(audio_tar)
    audio_pred = audio_format.pcm2float(audio_pred)
    audio_pred = audio_pred[:len(audio_tar)]
    #print(sig_name[l], ' : ', metrics.mean_squared_error(audio_tar, audio_pred))
    #results = measure_performance(audio_tar, audio_pred, name)
    #all_results.append(results)
    #plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec' + sig_name[l])
    #plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec' + sig_name[l])
    spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, sig_name[l] + name)

# with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
#     i = 0
#     for res in all_results:
#         print('\n', 'Sound', '  : ', sig_name[i], file=f)
#         i = i + 1
#         for key, value in res.items():
#             print('\n', key, '  : ', value, file=f)
