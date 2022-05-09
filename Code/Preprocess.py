# TODO add the following:
#   - positional encoding
#   - masking
#   - padding
#   - get batches
#   - scaling
#
# Much of the functionality is taken from: see: https://www.tensorflow.org/text/tutorials/transformer

import numpy as np
import tensorflow as tf
import copy


# ---------------------------------------------------------------------------------------------------------------------
# Positional encoding:
# ---------------------------------------------------------------------------------------------------------------------
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Masking:
# ---------------------------------------------------------------------------------------------------------------------
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ---------------------------------------------------------------------------------------------------------------------
# Custom Scaler:
#       Does exactly the same as a MinMaxScaler, but doesn't treat each time-step as a feature...
# ---------------------------------------------------------------------------------------------------------------------
class my_scaler():
    def __init__(self, feature_range=(0, 1)):
        super(my_scaler, self).__init__()
        self.max = feature_range[-1]
        self.min = feature_range[0]

    def fit(self, data):
        self.min_data = np.min(data)
        self.max_data = np.max(data)
        self.dtype = data.dtype

    def transform(self, data):
        X = copy.deepcopy(data)
        X_std = (X - self.min_data) / (self.max_data - self.min_data)
        return X_std * (self.max - self.min) + self.min

    def inverse_transform(self, data):
        X_scaled = copy.deepcopy(data)
        X_std = (X_scaled - self.min) / (self.max - self.min)
        X = X_std * (self.max_data - self.min_data) + self.min_data
        return X.astype(self.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Custom Scaler for three dimensional input (STFT):
#       Does exactly the same as a MinMaxScaler, but doesn't treat each time-step as a feature...
# ---------------------------------------------------------------------------------------------------------------------
class my_scaler_stft_OLD():
    def __init__(self, feature_range=(0, 1)):
        super(my_scaler_stft, self).__init__()
        self.min = feature_range[0]
        self.max = feature_range[-1]

    def fit(self, data):
        self.min_data = data.min(axis=1).min(axis=0)
        self.max_data = data.max(axis=1).max(axis=0)
        self.dtype = data.dtype

    def transform(self, data):
        X = copy.deepcopy(data)
        X_std = (X - self.min_data) / (self.max_data - self.min_data)
        return X_std * (self.max - self.min) + self.min

    def inverse_transform(self, data):
        X_scaled = copy.deepcopy(data)
        X_std = (X_scaled - self.min) / (self.max - self.min)
        X = X_std * (self.max_data - self.min_data) + self.min_data
        return X.astype(self.dtype)


class my_scaler_stft():
    def __init__(self, feature_range=(0, 1)):
        super(my_scaler_stft, self).__init__()
        self.min = feature_range[0]
        self.max = feature_range[-1]

    def fit(self, data):
        self.min_data = data.min()
        self.max_data = data.max()
        self.dtype = data.dtype

    def transform(self, data):
        X = copy.deepcopy(data)
        X_std = (X - self.min_data) / (self.max_data - self.min_data)
        return X_std * (self.max - self.min) + self.min

    def inverse_transform(self, data):
        X_scaled = copy.deepcopy(data)
        X_std = (X_scaled - self.min) / (self.max - self.min)
        X = X_std * (self.max_data - self.min_data) + self.min_data
        return X.astype(self.dtype)