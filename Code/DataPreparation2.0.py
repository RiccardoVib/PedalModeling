import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/VA'
    #data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA'
    #save_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    factor = 2#6#3
    #data_dir = kwargs.get('data_dir', '/Users/riccardosimionato/Datasets/VA')
    save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/VA/VA_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))


    #L = 32000000#se 48kHz
    L = 31000000

    #L = 32097856##10699286-100 #32097856#MAX=34435680
    #L = 5349643-100 #se 16kHz
    inp_collector, tar_collector, ratio_collector, threshold_collector = [], [], [], []
    inp_test, tar_test, ratio_test, threshold_test = [], [], [], []
    fs = 0

    inp_collector_never_seen, tar_collector_never_seen, ratio_collector_never_seen, threshold_collector_never_seen = [], [], [], []
    test_rec = False
    for file in file_dirs:

        if file == '/Users/riccardosimionato/Datasets/OD300/OD300_Tone_1_Drive_1_Mode_mid.wav':
            test_rec = True

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 2)
        ratio = metadata[1]
        threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        inp = audio_stereo[:L, 0]#.astype(np.float32)
        tar = audio_stereo[1:L+1, 1]#.astype(np.float32)

        inp_never_seen = audio_stereo[L:, 0]#.astype(np.float32)
        tar_never_seen = audio_stereo[L+1:, 1]#.astype(np.float32)

        ratio = str(ratio)
        if len(ratio) > 2:
            ratio = ratio[:2] + '.' + ratio[2:]

        #target is delayed by one sample due the system processing so
        #need to be moved
        #tar = tar[1:len(tar)]
        inp = audio_format.pcm2float(inp)
        tar = audio_format.pcm2float(tar)

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)

        inp_never_seen = signal.resample_poly(inp_never_seen, 1, factor)
        tar_never_seen = signal.resample_poly(tar_never_seen, 1, factor)

        ratio = float(ratio)
        threshold = float(threshold)
        #tar = np.pad(tar, (1, 0), mode='constant', constant_values=0)

        inp_collector_never_seen.append(inp_never_seen)
        tar_collector_never_seen.append(tar_never_seen)
        ratio_collector_never_seen.append(ratio)
        threshold_collector_never_seen.append(np.abs(threshold))

        if test_rec == False:

            # find
            sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
            sec = [32, 135, 238, 240.9, 308.7]
            sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
            start = np.zeros(len(sig_name), dtype=int)
            stop = np.zeros(len(sig_name), dtype=int)
            for l in range(len(sig_name)):
                start[l] = int(sec[l] * fs // factor)
                stop[l] = int(sec_end[l] * start[l])

            sweep_inp = inp[start[0]:stop[0]]
            guitar_inp = inp[start[1]:stop[1]]
            drumKick_inp = inp[start[2]:stop[2]]
            drumHH_inp = inp[start[3]:stop[3]]
            bass_inp = inp[start[4]:stop[4]]
            sweep_tar = tar[start[0]:stop[0]]
            guitar_tar = tar[start[1]:stop[1]]
            drumKick_tar = tar[start[2]:stop[2]]
            drumHH_tar = tar[start[3]:stop[3]]
            bass_tar = tar[start[4]:stop[4]]

            inp = np.concatenate((sweep_inp, guitar_inp, drumKick_inp, drumHH_inp, bass_inp))
            tar = np.concatenate((sweep_tar, guitar_tar, drumKick_tar, drumHH_tar, bass_tar))

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
            ratio_collector.append(ratio)
            threshold_collector.append(np.abs(threshold))
        else:

            inp_test.append(inp)
            tar_test.append(tar)
            ratio_test.append(ratio)
            threshold_test.append(np.abs(threshold))
            test_rec = False


    metadatas = {'ratio': ratio_collector, 'threshold': threshold_collector, 'samplerate': fs/factor}
    data = {'inp': inp_collector, 'tar': tar_collector}
    #
    # # train files
    # metadatas = {'ratio': ratio_collector, 'threshold': threshold_collector, 'samplerate': fs/factor}
    # data = {'inp': inp_collector, 'tar': tar_collector}
    # # test files
    # metadatas_test = {'ratio': ratio_test, 'threshold': threshold_test, 'samplerate': fs/factor}
    # data_test = {'inp': inp_test, 'tar': tar_test}
    # # never seen files
    # metadatas_never_seen = {'ratio': ratio_collector_never_seen, 'threshold': threshold_collector_never_seen, 'samplerate': fs/factor}
    # data_never_seen = {'inp': inp_collector_never_seen, 'tar': tar_collector_never_seen}
    #
    # # open a file, where you ant to store the data
    # # train files
    file_metadatas = open(os.path.normpath('/'.join([save_dir,'metadatas_test.pickle'])), 'wb')
    file_data = open(os.path.normpath('/'.join([save_dir,'data_test.pickle'])), 'wb')
    # # test files
    # file_metadatas_test = open(os.path.normpath('/'.join([save_dir,'metadatas_test_float.pickle'])), 'wb')
    # file_data_test = open(os.path.normpath('/'.join([save_dir,'data_test_float.pickle'])), 'wb')
    # # never seen files
    # file_metadatas_never_seen = open(os.path.normpath('/'.join([save_dir,'metadatas_never_seen_float.pickle'])), 'wb')
    # file_data_never_seen = open(os.path.normpath('/'.join([save_dir,'data_never_seen_float.pickle'])), 'wb')
    #
    # # dump information to that file
    # #train files
    pickle.dump(metadatas, file_metadatas)
    pickle.dump(data, file_data)
    # #test files
    # pickle.dump(metadatas_test, file_metadatas_test)
    # pickle.dump(data_test, file_data_test)
    # #never seen files
    # pickle.dump(metadatas_never_seen, file_metadatas_never_seen)
    # pickle.dump(data_never_seen, file_data_never_seen)
    #
    # # close the file
    file_metadatas.close()
    file_data.close()
    # file_metadatas_test.close()
    # file_data_test.close()
    # file_metadatas_never_seen.close()
    # file_data_never_seen.close()

if __name__ == '__main__':

    data_preparation()