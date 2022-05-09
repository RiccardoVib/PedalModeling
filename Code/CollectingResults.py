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
    #file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w1_limited.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w1.pickle'])), 'rb')
    data = pickle.load(file_data)
    x_test = data['x']
    #x_test = data['x_test']
    #y_test = data['y_test']
    #fs = data['fs']
    fs = 48000
    scaler = data['scaler']

    #create_ref()
    # Dense-----------------------------------------------------------------------------------
    if architecture=='dense':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Dense_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Dense_trials/DenseFeed_Testing_64_in1'
        name = 'Dense'
        #T=x_test.shape[1]
        audio_inp, audio_tar, audio_pred, fs = load_audio(data_dir)
        prediction_accuracy(audio_tar, audio_pred, audio_inp, fs, data_dir, name)

        data_dir_ref = '/Users/riccardosimionato/PycharmProjects/All_Results'
        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
        all_results = []
        for l in range(len(sig_name)):
            start = int(sec[l] * fs)
            stop = int(sec_end[l] * start)
            results = measure_performance(audio_tar[start:stop], audio_pred[start:stop], name)
            all_results.append(results)
            #spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, data_dir, sig_name[l] + name)

        # file_inp = glob.glob(os.path.normpath('/'.join([data_dir_ref, '_drumKick__inp.wav'])))
        # audio_inp = 0
        # for file in file_inp:
        #     fs, audio_inp = wavfile.read(file)
        # audio_inp = audio_format.pcm2float(audio_inp)
        #
        # N = len(audio_inp)
        # time = np.linspace(0, N / fs, num=N)
        # fig, ax = plt.subplots()
        # plt.title("Input - Time Domain")
        # ax.plot(time[6000:19200], audio_inp[6000:19200], 'g', label='Input')
        # ax.set(ylim=[-0.20, 0.20])
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Amplitude')
        # ax.legend()
        # fname = os.path.normpath(os.path.join(data_dir, name + '_drumKick__inp_time.png'))
        # fig.savefig(fname)
        #
        # N = len(audio_inp)
        # tukey = signal.windows.tukey(N, alpha=0.5)
        # fft_inp = fft.fftshift(fft.fft(audio_inp * tukey))[N // 2:]
        # fft_inp_smoth = mag_smoothing(fft_inp, 6)
        # freqs = fft.fftshift(fft.fftfreq(N) * fs)
        # freqs = freqs[N // 2:]
        # fig, ax = plt.subplots()
        # plt.title("Input - Frequency Domain")
        # ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_inp_smoth), np.max(np.abs(fft_inp)))), 'g--',
        #             label='Input')
        #
        # ax.set_xlabel('Frequency')
        # ax.set_ylabel('Magnitude (dB)')
        # ax.axis(xmin=20, xmax=22050)
        # ax.axis(ymin=-100, ymax=0)
        # ax.legend()
        # fname = os.path.normpath(os.path.join(data_dir, name + '_drumKick__inp_fft_compared.png'))
        # fig.savefig(fname)

        model = load_model_dense(1, units, drop, model_save_dir=data_dir)
        #measure_time(model, x_test, y_test, False, False, data_dir, fs, scaler, T)
        #
        with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
            i = 0
            for res in all_results:
                print('\n', 'Sound', '  : ', sig_name[i], file=f)
                i = i + 1
                for key, value in res.items():
                    print('\n', key, '  : ', value, file=f)
    # LSTM-----------------------------------------------------------------------------------
    if architecture == 'lstm':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/LSTM_Testing_64h'
        name = 'LSTM'
        #T=x_test.shape[1]
        audio_inp, audio_tar, audio_pred, fs = load_audio(data_dir)
        prediction_accuracy(audio_tar, audio_pred, audio_inp, fs, data_dir, name)

        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
        all_results = []
        for l in range(len(sig_name)):
            start = int(sec[l] * fs)
            stop = int(sec_end[l] * start)
            results = measure_performance(audio_tar[start:stop], audio_pred[start:stop], name)
            all_results.append(results)
            spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, data_dir, sig_name[l] + name)

        #model = load_model_lstm(T, units, drop, model_save_dir=data_dir)
        #measure_time(model, x_test, y_test, False, False, data_dir, fs, scaler, T)
        with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
            i=0
            for res in all_results:
                print('\n', 'Sound', '  : ', sig_name[i], file=f)
                i=i+1
                for key, value in res.items():
                    print('\n', key, '  : ', value, file=f)
    # --------------------------------------------------------------------------------------
    # change of dataset
    # --------------------------------------------------------------------------------------

    file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w2.pickle'])), 'rb')
    if w==4:
        file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w4.pickle'])), 'rb')
    elif w==8:
        file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w8.pickle'])), 'rb')
    elif w==16:
        #file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w16.pickle'])), 'rb')
        file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w16.pickle'])), 'rb')

    data = pickle.load(file_data)
    #x_test = data['x_test']
    #y_test = data['y_test']
    x_test = data['x']
    fs = data['fs']
    scaler = data['scaler']
    del data
    # LSTM_enc_dec----------------------------------------------------------------------------
    if architecture == 'lstm_enc_dec':
        data_dir_ref='/Users/riccardosimionato/PycharmProjects/All_Results'
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_trials/LSTM_enc_dec_32_32'
        name = 'LSTM_enc_dec'
        #T = x_test.shape[1]
        #enc_units = [units[0]]
        #dec_units = [units[1]]

        #encoder_model, decoder_model = load_model_lstm_enc_dec(T, enc_units, dec_units, 0., model_save_dir=data_dir)
        #model = [encoder_model, decoder_model]
        #time_s = measure_time(model, x_test, y_test, True, False, data_dir, fs, scaler, T)

        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]

        # for l in range(len(sig_name)):
        #
        #     start = int(sec[l] * fs)
        #     stop = int(sec_end[l] * start)
        #
        #     s1 = start // x_test.shape[1]
        #     s2 = stop // x_test.shape[1]
        #     hours = ((s2 - s1 * time_s / 60) / 60)
        #     days = hours / 24
        #     print('Number of samples to be generated: ', s2 - s1)
        #     print('Hours needed: ', hours)
        #     print('Days needed: ', days)
        #
        #     inferenceLSTM_enc_dec(data_dir=data_dir, fs=fs, x_test=x_test, y_test=y_test, scaler=scaler, start=start, stop=stop, name=sig_name[l], generate=True,
        #                           model=model, time=time_s, measuring=False)
        #


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
            #audio_inp = audio_inp[w:]
            #audio_tar = audio_tar[w:]
            audio_pred = audio_format.pcm2float(audio_pred)
            audio_pred = audio_pred[:len(audio_tar)]
            results = measure_performance(audio_tar, audio_pred, name)
            all_results.append(results)
            plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec' + sig_name[l])
            plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec' + sig_name[l])

        with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
            i=0
            for res in all_results:
                print('\n', 'Sound', '  : ', sig_name[i], file=f)
                i=i+1
                for key, value in res.items():
                    print('\n', key, '  : ', value, file=f)

    # LSTM_enc_dec_v2-------------------------------------------------------------------------
    if architecture == 'lstm_enc_dec_v2':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        data_dir_ref = '/Users/riccardosimionato/PycharmProjects/All_Results/TubeTech'
        name = 'LSTM_enc_dec_v2'
        T = x_test.shape[1]
        D = x_test.shape[2]
        enc_units = [units[0]]
        dec_units = [units[1]]

        model = load_model_lstm_enc_dec_v2(T=T, D=D, encoder_units=enc_units, decoder_units=dec_units, drop=drop, model_save_dir=data_dir)
        #time_s = measure_time(model=model, x_test=x_test, y_test=x_test, enc_dec=True, v2=True, data_dir=data_dir, fs=fs, scaler=scaler, T=T)
        #print(time_s)
        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
        for l in range(len(sig_name)):
            start = int(sec[l] * fs)
            stop = int(sec_end[l] * start)
            inferenceLSTM_enc_dec_v2(data_dir=data_dir, model=model, fs=fs, scaler=scaler, start=start, stop=stop, T=T, name=sig_name[l], generate=True, measuring=False)

        all_results = []
        for l in range(len(sig_name)):
            file_inp = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_inp.wav'])))
            file_tar = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_tar.wav'])))
            file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + '_pred.wav'])))
            for file in file_inp:
                fs, audio_inp = wavfile.read(file)
            for file in file_tar:
                fs, audio_tar = wavfile.read(file)
            for file in file_pred:
                _, audio_pred = wavfile.read(file)

            audio_inp = audio_format.pcm2float(audio_inp)
            audio_tar = audio_format.pcm2float(audio_tar)
            audio_inp = audio_inp[w:]
            audio_tar = audio_tar[w:]
            audio_pred = audio_format.pcm2float(audio_pred)
            plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
            plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
            results = measure_performance(audio_tar, audio_pred, name)
            all_results.append(results)

            spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, sig_name[l] + name)
            print(sig_name[l], ' : ', metrics.mean_squared_error(audio_tar, audio_pred))
        with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
            i=0
            for res in all_results:
                print('\n', 'Sound', '  : ', sig_name[i], file=f)
                i=i+1
                for key, value in res.items():
                    print('\n', key, '  : ', value, file=f)



if __name__ == '__main__':

    #retrive_info(architecture='dense', model_dir='DenseFeed_32_32', units=[32, 32], drop=0., w=1)
    #retrive_info(architecture='lstm', model_dir='LSTM_32_32_lin', units=[32, 32], drop=0., w=1)
    #retrive_info(architecture='lstm_enc_dec', model_dir='LSTM_enc_dec_32_32', units=[32, 32], drop=0., w=2)
    retrive_info(architecture='lstm_enc_dec_v2', model_dir='ED_mape', units=[64, 64], drop=0., w=16)
