from ProperEvaluationAllModels import load_audio, prediction_accuracy, measure_performance, measure_time, load_model_hybrid
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
from GetData_for_v2 import get_data



fs = 48000

model_dir='Hybrid_32_32'
units=[32, 32]
drop=0.
w=1

dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Hybrid/'
data_dir = os.path.normpath(os.path.join(dir, model_dir))
name = 'Hibrid_condition'
T=1

model = load_model_hybrid(T, units, drop, model_save_dir=data_dir)

sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
sec = [32, 135, 238, 240.9, 308.7]
sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]

for l in range(len(sig_name)):
    start = int(sec[l] * fs)
    stop = int(sec_end[l] * start)
    x_, y_, scaler = get_data(data_dir='../Files', start=start, stop=stop, T=T)
    predictions = model.predict(x_)

    predictions = np.array(predictions)
    predictions = scaler[0].inverse_transform(predictions)
    predictions = predictions.reshape(-1)
    pred_name = sig_name[l] + '_pred.wav'
    pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
    predictions = predictions.astype('int16')
    wavfile.write(pred_dir, int(fs), predictions)

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
    audio_inp = audio_inp[:-w]
    audio_tar = audio_tar[:-w]
    audio_inp = audio_format.pcm2float(audio_inp)
    audio_tar = audio_format.pcm2float(audio_tar)
    audio_pred = audio_format.pcm2float(audio_pred)
    plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM' + sig_name[l])
    plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM' + sig_name[l])
    results = measure_performance(audio_tar, audio_pred, name)
    all_results.append(results)

with open(os.path.normpath('/'.join([data_dir, 'performance_results_condition.txt'])), 'w') as f:
    i = 0
    for res in all_results:
        print('\n', 'Sound', '  : ', sig_name[i], file=f)
        i = i + 1
        for key, value in res.items():
            print('\n', key, '  : ', value, file=f)