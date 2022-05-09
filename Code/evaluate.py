import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fft
import os
import glob
import matplotlib.pyplot as plt
from TrainFunctionality import error_to_signal_ratio
from sklearn.metrics import r2_score
import librosa
import sklearn
import audio_format
import IPython.display as Ipd
#
def plot_result(data_dir, save):
    file_tar = glob.glob(os.path.normpath('/'.join([data_dir, '*_tar.wav'])))
    file_pred = glob.glob(os.path.normpath('/'.join([data_dir, '*_pred.wav'])))

    for file in file_tar:
        fs, audio_tar = wavfile.read(file)
        #audio_tar_ = librosa.load(file, 48000)
    for file in file_pred:
        _, audio_pred = wavfile.read(file)
        #audio_pred_ = librosa.load(file, 48000)

    #audio_tar = audio_tar.astype(np.float32)/32768
    #audio_pred = audio_pred[1600*0:].astype(np.float32)/32768

    audio_tar = audio_format.pcm2float(audio_tar)
    audio_pred = audio_format.pcm2float(audio_pred)
    #for index in range(len(audio_tar)//16):
    #    audio_tar = np.delete(audio_tar, index)
    #    index *= 16

    N = len(audio_pred)
    # = audio_pred[N//2:N//2+1600]
    #audio_tar = audio_tar[96000:384000]
    #audio_tar_ = audio_tar_[0]
    #audio_pred_ = audio_pred_[0]
    #audio_pred_ = audio_pred_[0: 1600]
    #audio_tar_ = audio_tar_[:len(audio_pred_)]

    print('ESR: %.4f' % error_to_signal_ratio(audio_tar, audio_pred))
    #r2_ = r2_score(audio_tar[:1600], audio_pred[:1600])
    #print(r2_)
    #audio_tar_dense = audio_tar_dense[1600:1600*2]
    #audio_pred_dense = audio_pred_dense[1600:1600*2]
    print('Coefficient of determination (r2 score): %.4f' % sklearn.metrics.r2_score(audio_tar, audio_pred))

    time = np.linspace(0, len(audio_tar) / fs, num=len(audio_tar))
    N = len(audio_tar)
    fs = 48000
    fft_tar = fft.fftshift(fft.fft(audio_tar))[N//2:]
    fft_pred = fft.fftshift(fft.fft(audio_pred))[N//2:]
    freqs = fft.fftshift(fft.fftfreq(N)*fs)
    freqs = freqs[N//2:]

    fig, ax = plt.subplots()
    plt.title("Target vs Prediction - Time Domain")
    ax.plot(time, audio_tar, 'b--', label='Target')
    ax.plot(time, audio_pred, 'r:', label='Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    fname = os.path.normpath('/'.join([data_dir, 'time.png']))

    #if save:
    #    fig.savefig(fname)

    fig, ax = plt.subplots(2,1)
    plt.suptitle("Target vs Prediction - Frequency Domain")
    ax[0].plot(freqs, 20*np.log10(np.divide(np.abs(fft_tar), np.max(np.abs(fft_tar)))), 'b', label='Target')
    ax[1].plot(freqs, 20*np.log10(np.divide(np.abs(fft_pred), np.max(np.abs(fft_pred)))), 'r', label='Prediction')
    ax[0].set_xlabel('Frequency')
    ax[1].set_xlabel('Frequency')
    ax[0].set_ylabel('Magnitude (dB)')
    ax[1].set_ylabel('Magnitude (dB)')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    fname = os.path.normpath('/'.join([data_dir, 'freq.png']))

    #if save:
        #fig.savefig(fname)

    #plt.close(fig)

    plt.plot(audio_tar)
    plt.title('v')
    plt.show()
    Ipd.display(Ipd.Audio(audio_tar, rate=48000))


if __name__ == '__main__':
    data_dir_ed = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_no_sig_Testing/WavPredictions'
    data_dir_eds = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_enc_dec_sig/WavPredictions'
    data_dir_LSTM = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_normal_no_sig/WavPredictions'
    data_dir_LSTMs = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_normal_sig/WavPredictions'
    data_dir_LSTMsT_1 = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_normal_sig_T1/WavPredictions'
    data_dir_Dense = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/DenseFeed_Testing/WavPredictions'
    data_dir_Dense_3 = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/DenseFeed_Testing_3/WavPredictions'
    DenseFeed_Testing_48_1_layer_sig = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/DenseFeed_Testing_48_1_layer_sig/WavPredictions'
    DenseFeed_Testing_getdata2_48_input1 = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/DenseFeed_Testing_getdata2_48_input1/WavPredictions'
    LSTM_enc_dec = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec/WavPredictions'
    LSTM_enc_dec_v2_2 = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_v2_16/WavPredictions'



    #plot_result(data_dir=data_dir_ed, save=True)
    #plot_result(data_dir=data_dir_eds, save=True)
    #plot_result(data_dir=data_dir_LSTM, save=True)
    #plot_result(data_dir=data_dir_LSTMs, save=True)
    #plot_result(data_dir=data_dir_LSTMsT_1, save=True)
    #plot_result(data_dir=DenseFeed_Testing_48_1_layer_sig, save=True)
    #plot_result(data_dir=DenseFeed_Testing_getdata2_48_input1, save=True)
    #plot_result(data_dir=data_dir_Dense, save=True)#esr:0.5543166281246964
    #plot_result(data_dir=data_dir_Dense_3, save=True)#esr:5568089993155195

    # LSTM_enc_dec
    plot_result(data_dir=LSTM_enc_dec, save=True)
    # LSTM_enc_dec_v2
    #plot_result(data_dir=LSTM_enc_dec_v2_2, save=True)