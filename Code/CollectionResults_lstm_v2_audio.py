from ProperEvaluationAllModels import measure_performance, plot_time, plot_fft
from ProperEvaluationAllModels import load_model_lstm_enc_dec_v2, inferenceLSTM_enc_dec_v2
import os
import pickle
from scipy.io import wavfile
import numpy as np
from GetData_onlyTestSamples import get_data
import glob
import audio_format

def inferenceLSTM_enc_dec_v2(data_dir, model, fs, T, sets, name):
    x_, y_ , scaler = get_data(data_dir='../Files', sets=sets, T=T)
    for i in range(5):
        signal_i = x_[i]
        predictions = model.predict([signal_i[:, :-1, :], signal_i[:, -1, 0].reshape(signal_i.shape[0], 1, 1)])
        predictions = np.array(predictions)
        predictions = scaler[0].inverse_transform(predictions)
        predictions = predictions.reshape(-1)
        pred_name = name[i] + str(sets) + '_pred.wav'
        pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        predictions = predictions.astype('int16')
        wavfile.write(pred_dir, int(fs), predictions)


def retrive_info(model_dir, units, drop, w):

    # --------------------------------------------------------------------------------------
    # change of dataset
    # --------------------------------------------------------------------------------------
    T = w
    fs = 48000

    dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/'
    data_dir = os.path.normpath(os.path.join(dir, model_dir))
    data_dir_ref='/Users/riccardosimionato/PycharmProjects/All_Results'
    name = 'LSTM_enc_dec_v2'
    enc_units = [units[0]]
    dec_units = [units[1]]

    model = load_model_lstm_enc_dec_v2(T=T, encoder_units=enc_units, decoder_units=dec_units, drop=drop, model_save_dir=data_dir)

    sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
    sets = 'test'
    inferenceLSTM_enc_dec_v2(data_dir=data_dir, model=model, fs=fs, sets=sets, T=T, name=sig_name)
    #
    # all_results = []
    # for l in range(len(sig_name)):
    #     file_inp = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_inp.wav'])))
    #     file_tar = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + str(sets) + '_tar.wav'])))
    #     file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + str(sets) + '_pred.wav'])))
    #     for file in file_inp:
    #         fs, audio_inp = wavfile.read(file)
    #     for file in file_tar:
    #         fs, audio_tar = wavfile.read(file)
    #     for file in file_pred:
    #         _, audio_pred = wavfile.read(file)
    #
    #     audio_inp = audio_format.pcm2float(audio_inp)
    #     audio_tar = audio_format.pcm2float(audio_tar)
    #     audio_inp = audio_inp[w:]
    #     audio_tar = audio_tar[w:]
    #     audio_pred = audio_format.pcm2float(audio_pred)
    #     results = measure_performance(audio_tar, audio_pred, name)
    #     all_results.append(results)
    #     plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
    #     plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
    #     #spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, sig_name[l] + name)
    #     #print(sig_name[l], ' : ', metrics.mean_squared_error(audio_tar, audio_pred))
    # with open(os.path.normpath('/'.join([data_dir, str(sets) + 'performance_results.txt'])), 'w') as f:
    #     i=0
    #     for res in all_results:
    #         print('\n', 'Sound', '  : ', sig_name[i], file=f)
    #         i=i+1
    #         for key, value in res.items():
    #             print('\n', key, '  : ', value, file=f)



if __name__ == '__main__':

    retrive_info(model_dir='LSTM_enc_dec_v2_16', units=[8, 8], drop=0., w=16)
