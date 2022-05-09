import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
from scipy.io import wavfile

dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/LSTM_enc_dec_v2_16'

file_pred = glob.glob(os.path.normpath('/'.join([dir, '_sweep__pred.wav'])))
file_tar = glob.glob(os.path.normpath('/'.join([dir, '_sweep__tar.wav'])))
file_inp = glob.glob(os.path.normpath('/'.join([dir, '_sweep__inp.wav'])))
amp = 2 * np.sqrt(2)
for file in file_tar:
    fs, sweep_tar = wavfile.read(file)
for file in file_pred:
    fs, sweep_pred = wavfile.read(file)
for file in file_inp:
    fs, sweep_inp = wavfile.read(file)

N = len(sweep_pred)
f, t, Zxx = signal.stft(sweep_pred, fs=fs)#, nfft=N//12, nperseg=N//12)
#plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# fig, ax = plt.subplots()
# plt.title("Aliasing")
# ax.plot(f, np.abs(Zxx[:,60]), 'b--')
# ax.set_xlabel('Frequency')
# ax.set_ylabel('Magnitude')
# ax.legend()
# plt.show()