import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fft
import os
import glob
import matplotlib.pyplot as plt
from TrainFunctionality import error_to_signal_ratio
from GetData_for_v2 import get_data
import sklearn
import audio_format
import time
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
from InferenceLSTM import predict_sequence
from mag_smoothing import mag_smoothing
from librosa import display
import pickle

def predict_sinusoids(f, fs, model):
    fs = 48000
    dt = 1 / fs
    StopTime = 0.25
    t = np.linspace(start=0, stop=StopTime-dt, num=dt)
    sinusoid = np.sin(2 * np.pi * f * t)
    return sinusoid
#plotting

def spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, name):
    vmin = -40
    f, t, Zxx = signal.stft(audio_tar, fs=fs)
    fig, ax = plt.subplots()
    plt.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)), vmin=vmin, vmax=0, shading='gouraud')
    #plt.imshow(mat, origin="lower", cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.legend()
    #plt.show()
    fname = os.path.normpath(os.path.join(data_dir, name + 'tar_spectrogram.png'))
    fig.savefig(fname)

    f, t, Zxx = signal.stft(audio_pred, fs=fs)
    fig, ax = plt.subplots()
    plt.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)), vmin=vmin, vmax=0, shading='gouraud')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    fname = os.path.normpath(os.path.join(data_dir, name + 'pred_spectrogram.png'))
    fig.savefig(fname)

    f, t, Zxx = signal.stft(audio_inp, fs=fs)
    fig, ax = plt.subplots()
    plt.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)), vmin=vmin, vmax=0, shading='gouraud')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    fname = os.path.normpath(os.path.join(data_dir, name + 'inp_spectrogram.png'))
    fig.savefig(fname)

def plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, name):
    N = len(audio_tar)
    time = np.linspace(0, N / fs, num=N)
    tukey = signal.windows.tukey(N, alpha=0.1)
    #audio_tar_C = np.convolve(audio_tar, tukey, 'valid')
    #audio_pred_C = np.convolve(audio_pred, tukey, 'valid')
    if name == 'Dense_drumKick_' or name == 'LSTM_drumKick_' or name == 'LSTM_enc_dec_drumKick_' or name == 'LSTM_enc_dec_v2_drumKick_':
        fig, ax = plt.subplots()
        plt.title("Target vs Prediction - Time Domain")
        ax.plot(time[6000:19200], audio_tar[6000:19200], 'b--', label='Target')
        ax.plot(time[6000:19200], audio_pred[6000:19200], 'r:', label='Prediction')
        ax.set(ylim=[-0.20, 0.20])
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        #display.waveshow(audio_tar[6000:19200], sr=fs, ax=ax)
        #display.waveshow(audio_pred[6000:19200], sr=fs, ax=ax)
        #plt.show()
    else:
        fig, ax = plt.subplots()
        plt.title("Target vs Prediction - Time Domain")
        #ax.plot(time, audio_tar, 'b--', label='Target')
        #ax.plot(time, audio_pred, 'r:', label='Prediction')
        #ax.plot(time, tukey, 'r:', label='Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set(ylim=[-0.3, 0.3])#, xlim=[0,len(audio_tar)])
        display.waveshow(audio_inp, sr=fs, ax=ax, label='Input')
        display.waveshow(audio_tar, sr=fs, ax=ax, label='Target', alpha=0.7)
        display.waveshow(audio_pred, sr=fs, ax=ax, label='Prediction', alpha=0.5)
        ax.label_outer()
        ax.legend()
        #plt.show()

    fname = os.path.normpath(os.path.join(data_dir, name + '_time.png'))
    fig.savefig(fname)
def plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, name):
    N = len(audio_tar)
    tukey = signal.windows.tukey(N, alpha=0.5)
    fft_inp = fft.fftshift(fft.fft(audio_inp * tukey))[N // 2:]
    fft_tar = fft.fftshift(fft.fft(audio_tar*tukey))[N // 2:]
    fft_pred = fft.fftshift(fft.fft(audio_pred*tukey))[N // 2:]
    fft_inp_smoth = mag_smoothing(fft_inp, 6)
    fft_tar_smoth = mag_smoothing(fft_tar, 6)
    fft_pred_smoth = mag_smoothing(fft_pred, 6)
    freqs = fft.fftshift(fft.fftfreq(N) * fs)
    freqs = freqs[N // 2:]
    # fig, ax = plt.subplots(2,1)
    # plt.suptitle("Target vs Prediction - Frequency Domain")
    # ax[0].semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_tar_smoth), np.max(np.abs(fft_tar)))), 'b--', label='Target')#, linewidth=0.5, markersize=12)
    # ax[1].semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_pred_smoth), np.max(np.abs(fft_pred)))), 'r--', label='Prediction')#, linewidth=0.1, markersize=12)
    # ax[0].set_xlabel('Frequency')
    # ax[1].set_xlabel('Frequency')
    # ax[0].set_ylabel('Magnitude (dB)')
    # ax[1].set_ylabel('Magnitude (dB)')
    # ax[0].axis(xmin=20,xmax=22050)
    # ax[1].axis(xmin=20,xmax=22050)
    # ax[0].axis(ymin=-100,ymax=0)
    # ax[1].axis(ymin=-100,ymax=0)
    # ax[0].legend()
    # ax[1].legend()
    #plt.show()

    #fname = os.path.normpath(os.path.join(data_dir, name + '_fft.png'))
    #fig.savefig(fname)

    fig, ax = plt.subplots()
    plt.title("Target vs Prediction - Frequency Domain")


    if name == 'Dense_drumKick_' or name == 'LSTM_drumKick_' or name == 'LSTM_enc_dec_drumKick_' or name == 'LSTM_enc_dec_v2_drumKick_':
        ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_tar_smoth), np.max(np.abs(fft_tar)))), 'b--',
                    label='Target')
        ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_pred_smoth), np.max(np.abs(fft_pred)))), 'r--',
                    label='Prediction')
    else:
        ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_inp_smoth), np.max(np.abs(fft_inp)))), 'b--',
                    label='Input')
        ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_tar_smoth), np.max(np.abs(fft_tar)))), 'y--',
                    label='Target')
        ax.semilogx(freqs, 20 * np.log10(np.divide(np.abs(fft_pred_smoth), np.max(np.abs(fft_pred)))), 'g--',
                    label='Prediction')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.axis(xmin=20,xmax=22050)
    ax.axis(ymin=-100,ymax=0)
    ax.legend()
    fname = os.path.normpath(os.path.join(data_dir, name + '_fft_compared.png'))
    fig.savefig(fname)
#measuring
def measure_time(model, x_test, y_test, enc_dec, v2, data_dir, fs, scaler, T):

    if enc_dec:
        if v2:
            start = time.time()
            inferenceLSTM_enc_dec_v2(data_dir=data_dir, model=model, fs=fs, scaler=scaler, T=T, start=0, stop=17, name='test_time', generate=False, measuring=True)
            end = time.time()
            time_s = (end - start)#/x_test.shape[0]
            #print('Time: %.6f' % time_s)
        else:
            encoder_model = model[0]
            decoder_model = model[1]
            start = time.time()
            inferenceLSTM_enc_dec(data_dir=data_dir, fs=fs, x_test=x_test, y_test=y_test, scaler=scaler, start=0, stop=1, name='test_time', generate=False, model=model, time=0, measuring=True)
            end = time.time()
            time_s = end - start
            print('Time: %.6f' % time_s)
    else:
        start = time.time()
        model.predict(x_test)
        end = time.time()
        time_s = (end - start)/x_test.shape[0]
        print('Time: %.6f' % time_s)

    with open(os.path.normpath('/'.join([data_dir, 'performance_time.txt'])), 'w') as f:
        print('Time: %.6f' % time_s, file=f)
    return time_s
def measure_performance(audio_tar, audio_pred, name):

    ESR = error_to_signal_ratio(audio_tar, audio_pred)
    r2 = sklearn.metrics.r2_score(audio_tar, audio_pred)
    print('ESR: %.6f' % ESR)
    print('Coefficient of determination (r2 score): %.6f' % r2)

    results = {
        'Model': name,
        'ESR': ESR,
        'r2': r2
    }
    return results
#loading
def load_audio(data_dir):
    data_dir = os.path.normpath(os.path.join(data_dir, 'WavPredictions'))
    file_tar = glob.glob(os.path.normpath('/'.join([data_dir, '*_tar.wav'])))
    file_pred = glob.glob(os.path.normpath('/'.join([data_dir, '*_pred.wav'])))
    file_inp = glob.glob(os.path.normpath('/'.join([data_dir, '*_inp.wav'])))

    for file in file_inp:
        fs, audio_inp = wavfile.read(file)
    for file in file_tar:
        fs, audio_tar = wavfile.read(file)

    for file in file_pred:
        _, audio_pred = wavfile.read(file)

    audio_inp = audio_format.pcm2float(audio_inp)
    audio_tar = audio_format.pcm2float(audio_tar)
    audio_pred = audio_format.pcm2float(audio_pred)
    return audio_inp, audio_tar, audio_pred, fs
def load_ref(data_dir = '/Users/riccardosimionato/Datasets/VA'):

    L = 31000000
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TubeTech_333_-30.wav'])))
    #file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TubeTech_466_-10.wav'])))
    #file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TubeTech_733_-40.wav'])))
    for file in file_dirs:
        fs, audio_stereo = wavfile.read(file)  # fs= 96,000 Hz
        inp = audio_stereo[:L, 0].astype(np.float32)
        tar = audio_stereo[1:L + 1, 1].astype(np.float32)

        inp = signal.resample_poly(inp, 1, 2)
        tar = signal.resample_poly(tar, 1, 2)
        fs = fs // 2
    return inp, tar, fs
#test samples
def prediction_accuracy(tar, pred, inp, fs, data_dir, name):
    sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
    sec = [32, 135, 238, 240.9, 308.7]
    sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
    for l in range(len(sig_name)):
        start = int(sec[l] * fs)
        stop = int(sec_end[l] * start)
        plot_time(tar[start:stop], pred[start:stop], inp[start:stop], fs, data_dir, name + sig_name[l])
        plot_fft(tar[start:stop], pred[start:stop], inp[start:stop], fs, data_dir, name + sig_name[l])
        pred_name = name + sig_name[l] + '_pred.wav'
        tar_name = name + sig_name[l] + '_tar.wav'
        pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        wavfile.write(pred_dir, int(fs), pred[start:stop])
        wavfile.write(tar_dir, int(fs), tar[start:stop])
def create_ref(data_dir='/Users/riccardosimionato/PycharmProjects/All_Results'):
    inp, tar, fs = load_ref()
    sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
    sec = [32, 135, 238, 240.9, 308.7]
    sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
    for l in range(len(sig_name)):
        start = int(sec[l] * fs)
        stop = int(sec_end[l] * start)
        inp_ = inp[start:stop].astype('int16')
        tar_ = tar[start:stop].astype('int16')
        inp_name = sig_name[l] + '_inp.wav'
        tar_name = sig_name[l] + '_tar.wav'
        inp_dir = os.path.normpath(os.path.join(data_dir, inp_name))
        tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        wavfile.write(inp_dir, int(fs), inp_)
        wavfile.write(tar_dir, int(fs), tar_)
#loading models
def load_model_dense(T,units,drop,model_save_dir):

    inputs = Input(shape=(T,3), name='input')
    first_unit = units.pop(0)
    if len(units) > 0:
        last_unit = units.pop()
        outputs = Dense(first_unit, name='Dense_0')(inputs)
        for i, unit in enumerate(units):
            outputs = Dense(unit, name='Dense_' + str(i + 1))(outputs)
        outputs = Dense(last_unit, activation='tanh', name='Dense_Fin')(outputs)
    else:
        outputs = Dense(first_unit, activation='tanh', name='Dense')(inputs)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    final_outputs = Dense(1, activation='tanh', name='DenseLay')(outputs)
    model = Model(inputs, final_outputs)

    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, 'Checkpoints', 'best'))
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(latest)
    else:
        raise ValueError('Something wrong!')
    return model
def load_model_lstm(T,units,drop, model_save_dir):

    inputs = Input(shape=(T,3), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(inputs)

    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    outputs = Dense(1, activation='tanh', name='DenseLay')(outputs)
    model = Model(inputs, outputs)

    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, 'Checkpoints', 'best'))
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(latest)
    else:
        raise ValueError('Something wrong!')
    return model

def load_model_hybrid(T,units,drop, model_save_dir):
    inputs = Input(shape=(T, 3), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs, state_h, state_c = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs = LSTM(last_unit_encoder, name='LSTM_EnFin')(outputs)
    else:
        outputs = LSTM(first_unit_encoder, name='LSTM_En')(inputs)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    outputs = Dense(32, activation='tanh', name='NonlinearDenseLay')(outputs)
    outputs = Dense(1, name='DenseLay')(outputs)
    model = Model(inputs, outputs)

    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, 'Checkpoints', 'best'))
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(latest)
    else:
        raise ValueError('Something wrong!')
    return model

def load_model_lstm_enc_dec(T, encoder_units, decoder_units,drop, model_save_dir):
    encoder_inputs = Input(shape=(T, 3), name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(T - 1, 1), name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        decoder_lstm = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De0', dropout=drop)
        outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            decoder_lstm = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1), dropout=drop)
            outputs, _, _ = decoder_lstm(outputs)
        decoder_lstm = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin', dropout=drop)
        outputs, _, _ = decoder_lstm(outputs)
    else:
        decoder_lstm = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De', dropout=drop)
        outputs, _, _ =decoder_lstm(decoder_inputs, initial_state=encoder_states)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_dense = Dense(1, activation='sigmoid', name='DenseLay')
    decoder_output = decoder_dense(outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_output)

    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, 'Checkpoints', 'best'))
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(latest)
    else:
        raise ValueError('Something wrong!')

    # INFERENCE
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(first_unit_decoder,))
    decoder_state_input_c = Input(shape=(first_unit_decoder,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model
def load_model_lstm_enc_dec_v2(T, D, encoder_units, decoder_units, drop, model_save_dir):
    encoder_inputs = Input(shape=(T-1,D), name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(1,1), name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De0', dropout=drop)(decoder_inputs,
                                                                                   initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1), dropout=drop)(outputs)
        outputs, _, _ = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin', dropout=drop)(outputs)
    else:
        outputs, _, _ = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De', dropout=drop)(
                                                                                        decoder_inputs,
                                                                                        initial_state=encoder_states)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_outputs = Dense(1, activation='sigmoid', name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, 'Checkpoints', 'best'))
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(latest)
    else:
        raise ValueError('Something wrong!')
    return model
#inference
def inferenceLSTM_enc_dec(data_dir, fs, x_test, y_test, scaler, start, stop, name, generate, model, time, measuring=False):

    s1 = start // x_test.shape[1]
    s2 = stop // x_test.shape[1]

    #last_prediction = y_test[s1, 0]
    #predictions = [last_prediction]
    output_dim = y_test.shape[1]-1
    n_steps = y_test.shape[1]-1
    encoder_model = model[0]
    decoder_model = model[1]
    predictions = []
    if measuring:
        last_prediction = 0
        predict_sequence(encoder_model, decoder_model, x_test[0, :, :], n_steps, output_dim, last_prediction, x_test.shape[1])
    else:
        for b in range(s1, s2):
            last_prediction = y_test[b, 0]
            last_prediction_ = np.array(y_test[b, 0]).reshape(1,1)

            #predictions.append(last_prediction)
            out, last_prediction = predict_sequence(encoder_model, decoder_model, x_test[b, :, :], n_steps, output_dim,
                                                    last_prediction, x_test.shape[1])

            out_ = np.concatenate((last_prediction_, out))
            predictions.append(out_)
    #     for b in range(s1, s2):
    #         out, last_prediction = predict_sequence(encoder_model, decoder_model, x_test[b, :, :], n_steps, output_dim, last_prediction)
    #         predictions.append(out)
    #         x_ = np.zeros((1, 2, 3))
    #         x_[0, 0, 0] = x_test[b, 1, 0]
    #         x_[0, 1, 0] = x_test[b + 1, 0, 0]
    #         x_[0, :, 1] = x_test[b, 0, 1]
    #         x_[0, :, 2] = x_test[b, 0, 2]
    #         out, last_prediction = predict_sequence(encoder_model, decoder_model, x_, n_steps, output_dim, last_prediction)
    #         predictions.append(out)
    #

    if generate:
        predictions = np.array(predictions)
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_test[s1:s2, :, 0])
        y_gen = scaler[0].inverse_transform(y_test[s1:s2])

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = name + '_pred.wav'
        inp_name = name + '_inp.wav'
        tar_name = name + '_tar.wav'

        pred_dir = os.path.normpath(os.path.join(data_dir, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(data_dir, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(data_dir, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, int(fs), predictions)
        wavfile.write(inp_dir, int(fs), x_gen)
        wavfile.write(tar_dir, int(fs), y_gen)
def inferenceLSTM_enc_dec_v2(data_dir, model, fs, scaler, T, start, stop, name, generate, measuring):

    x_, y_ , scaler = get_data(data_dir='../Files', start=start, stop=stop, T=T)
    # data_ = '../Files'
    # file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w16.pickle'])), 'rb')
    # data = pickle.load(file_data)
    # x_ = data['x']
    # fs = data['fs']
    # scaler = data['scaler']
    if measuring:
        start = time.time()
        predictions = model.predict([x_[:, :-1, :].reshape(x_.shape[0], 15, 3), x_[:, -1, 0].reshape(x_.shape[0], 1, 1)])
        end = time.time()
        time_s = (end - start)/x_.shape[0]
        print('Time: %.6f' % time_s)
    else:
        predictions = model.predict([x_[:, :-1, :], x_[:, -1, 0].reshape(x_.shape[0], 1, 1)])

    if generate:
        predictions = np.array(predictions)
        if scaler is not None:
            predictions = scaler[0].inverse_transform(predictions)
        predictions = predictions.reshape(-1)
        pred_name = name + '_pred.wav'
        pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))
        predictions = predictions.astype('int16')
        wavfile.write(pred_dir, int(fs), predictions)