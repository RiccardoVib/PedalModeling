import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(cond_number):
    data_dir = 'C:/Users/riccarsi/Documents/GitHub/OD300/Cond_2'
    save_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    factor = 2
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    #L = 32000000#se 48kHz
    L = 31000000
    if cond_number == 3:
        test_file = 'OD300_Tone_1_Drive_1_Mode_mid.wav'
    elif cond_number == 2:
        test_file = 'OD300_Tone_1_Drive_1_Mode_ds.wav'
    else:
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
            #find
            # sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
            # sec = [32, 135, 238, 240.9, 308.7]
            # sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
            # start = np.zeros(len(sig_name), dtype=int)
            # stop = np.zeros(len(sig_name), dtype=int)
            # for l in range(len(sig_name)):
            #     start[l] = int(sec[l] * fs // factor)
            #     stop[l] = int(sec_end[l] * start[l])
            #
            # sweep_inp = inp[start[0]:stop[0]]
            # guitar_inp = inp[start[1]:stop[1]]
            # drumKick_inp = inp[start[2]:stop[2]]
            # drumHH_inp = inp[start[3]:stop[3]]
            # bass_inp = inp[start[4]:stop[4]]
            # sweep_tar = tar[start[0]:stop[0]]
            # guitar_tar = tar[start[1]:stop[1]]
            # drumKick_tar = tar[start[2]:stop[2]]
            # drumHH_tar = tar[start[3]:stop[3]]
            # bass_tar = tar[start[4]:stop[4]]
            #
            # inp = np.concatenate((sweep_inp, guitar_inp, drumKick_inp, drumHH_inp, bass_inp))
            # tar = np.concatenate((sweep_tar, guitar_tar, drumKick_tar, drumHH_tar, bass_tar))

            # # remove
            # inp1 = inp[:start[0]]
            # inp2 = inp[stop[0] + 1:start[1]]
            # inp3 = inp[stop[1] + 1:start[2]]
            # inp4 = inp[stop[2] + 1:start[3]]
            # inp5 = inp[stop[3] + 1:start[4]]
            # inp6 = inp[stop[4] + 1:]
            #
            # inp = np.concatenate((inp1, inp2, inp3, inp4, inp5, inp6))
            # tar1 = tar[:start[0]]
            # tar2 = tar1[stop[0] + 1:start[1]]
            # tar3 = tar1[stop[1] + 1:start[2]]
            # tar4 = tar1[stop[2] + 1:start[3]]
            # tar5 = tar1[stop[3] + 1:start[4]]
            # tar6 = tar1[stop[4] + 1:]
            # tar = np.concatenate((tar1, tar2, tar3, tar4, tar5, tar6))

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

    if cond_number==3:
        metadatas = {'tone': tone_collector, 'drive': drive_collector, 'mode': mode_collector, 'samplerate': fs/factor}
        data = {'inp': inp_collector, 'tar': tar_collector}

        metadatas_test = {'tone': tone_test, 'drive': drive_test, 'mode': mode_test, 'samplerate': fs/factor}
        data_test = {'inp': inp_test, 'tar': tar_test}

    elif cond_number==2:
        metadatas = {'tone': tone_collector, 'drive': drive_collector, 'samplerate': fs/factor}
        data = {'inp': inp_collector, 'tar': tar_collector}

        metadatas_test = {'tone': tone_test, 'drive': drive_test, 'samplerate': fs/factor}
        data_test = {'inp': inp_test, 'tar': tar_test}
    else:
        metadatas = {'tone': tone_collector, 'samplerate': fs/factor}
        data = {'inp': inp_collector, 'tar': tar_collector}

        metadatas_test = {'tone': tone_test, 'samplerate': fs/factor}
        data_test = {'inp': inp_test, 'tar': tar_test}

    # open a file, where you ant to store the data
    # train files
    file_metadatas = open(os.path.normpath('/'.join([save_dir, 'metadatasOD300_cond3.pickle'])), 'wb')
    file_data = open(os.path.normpath('/'.join([save_dir, 'dataOD300_cond3.pickle'])), 'wb')
    # test files
    file_metadatas_test = open(os.path.normpath('/'.join([save_dir, 'metadatas_test_OD300_cond3.pickle'])), 'wb')
    file_data_test = open(os.path.normpath('/'.join([save_dir, 'data_test_OD300_cond3.pickle'])), 'wb')

    # dump information to that file
    pickle.dump(metadatas, file_metadatas)
    pickle.dump(data, file_data)
    # test files
    pickle.dump(metadatas_test, file_metadatas_test)
    pickle.dump(data_test, file_data_test)

    file_metadatas.close()
    file_data.close()
    file_metadatas_test.close()
    file_data_test.close()

if __name__ == '__main__':

    data_preparation(cond_number=3)