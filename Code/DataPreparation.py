import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation():
    #data_dir = 'C:/Users/riccarsi/Documents/GitHub/OD300/Cond_2'
    data_dir = '/Users/riccardosimionato/Datasets/OD300'
    save_dir = '../Files'
    factor = 2
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    #L = 32000000#se 48kHz
    L = 31000000
    test_file = 'OD300_Tone_1_Drive_1_Mode_ds.wav'

    inp_collector, tar_collector, tone_collector, drive_collector, mode_collector = [], [], [], [], []
    inp_test, tar_test, tone_test, drive_test, mode_test = [], [], [], [], []
    fs = 0
    test_rec = False
    for file in file_dirs:

        filename = os.path.split(file)[-1]

        if filename == test_file:
            test_rec = True

        metadata = filename.split('_', 7)
        tone = metadata[2]
        drive = metadata[4]
        mode = metadata[-1].replace('.wav', '')

        if mode == 'ds':
            mode = 0
        elif mode == 'mid':
            mode = 0.5
        elif mode == 'od':
            mode == 1

        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        inp = audio_stereo[:L, 0]
        tar = audio_stereo[1:L+1, 1]
        #target is delayed by one sample due the system processing so

        inp = audio_format.pcm2float(inp)
        tar = audio_format.pcm2float(tar)

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)

        if test_rec == False:

            inp_collector.append(inp)
            tar_collector.append(tar)
            tone_collector.append(tone)
            drive_collector.append(drive)
            mode_collector.append(mode)
        else:

            inp_test.append(inp)
            tar_test.append(tar)
            tone_test.append(tone)
            drive_test.append(drive)
            mode_test.append(mode)
            test_rec = False

    data = {'inp': inp_collector, 'tar': tar_collector, 'tone': tone_collector, 'drive': drive_collector, 'mode': mode_collector, 'samplerate': fs/factor}
    data_test = {'inp': inp_test, 'tar': tar_test, 'tone': tone_test, 'drive': drive_test, 'mode': mode_test, 'samplerate': fs/factor}

    # open a file, where you ant to store the data
    file_data = open(os.path.normpath('/'.join([save_dir, 'data_OD300.pickle'])), 'wb')
    file_data_test = open(os.path.normpath('/'.join([save_dir, 'data_test_OD300.pickle'])), 'wb')

    pickle.dump(data, file_data)
    pickle.dump(data_test, file_data_test)

    file_data.close()
    file_data_test.close()

if __name__ == '__main__':

    data_preparation()