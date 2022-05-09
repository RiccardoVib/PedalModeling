import pickle
import os
import numpy as np

data_dir = '../Files'

file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16.pickle'])), 'rb')

Z = pickle.load(file_data)

x = Z['x']