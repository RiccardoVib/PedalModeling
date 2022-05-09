import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle

data_dir = '/Users/riccardosimionato/Datasets/VA'

factor = 2#6#3
file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TubeTech_866_-40.wav'])))
for file in file_dirs:

    fs, audio_stereo = wavfile.read(file)  # fs= 96,000 Hz
    inp = audio_stereo[:,0].astype(np.float32)

    inp = signal.resample_poly(inp, 1, factor)

    inp = inp.astype('int16')

    inp_dir = os.path.normpath(os.path.join('inp.wav'))
    wavfile.write(inp_dir, 48000, inp)