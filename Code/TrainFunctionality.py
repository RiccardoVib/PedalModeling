import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from keras import backend as K


def custom_loss_function(y_true, y_pred):
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    #squared_difference = tf.square(tf.abs(y_true - y_pred))
    #loss = squared_difference/tf.square(tf.abs(y_true))
    squared_difference_dc = tf.square(tf.abs(y_true - y_pred))/(y_true.shape[0])
    mse_dc = squared_difference_dc/(tf.square(tf.abs(y_true))/y_true.shape[0])
    return mse+mse_dc#tf.reduce_mean(squared_difference, axis=-1)


def ESR_loss_function(y_true, y_pred):
    return tf.divide(K.sum(K.square(y_pred - y_true)), K.sum(K.square(y_true)))

def FFT_loss_function(y_true, y_pred):
    Y_true = tf.signal.rfft(tf.reshape(y_true, [-1]))
    Y_pred = tf.signal.rfft(tf.reshape(y_pred, [-1]))
    return tf.keras.metrics.mean_absolute_error(tf.math.real(Y_true), tf.math.real(Y_pred))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def get_batches(x, y, b_size, shuffle=True, seed=99):
    np.random.seed(seed)
    indxs = np.arange(tf.shape(x)[0])
    if shuffle:
        np.random.shuffle(indxs)

    def divide_chunks(l, n):
        # looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    x_b, y_b = [], []
    indxs = divide_chunks(indxs, b_size)

    for indx_batch in indxs:
        # if len(indx_batch) != b_size:
        #     continue
        x_b.append(x[indx_batch])
        y_b.append(y[indx_batch])

    return x_b, y_b


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def squared_error(ys_orig, ys_pred):
    return sum((ys_pred - ys_orig) * (ys_pred - ys_orig))


def coefficient_of_determination(ys_orig, ys_pred):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_pred)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


def error_to_signal_ratio(ys_orig, ys_pred):
    num, den = 0,0
    for n in range(len(ys_orig)):
        num += (ys_orig[n] - ys_pred[n])**2
        den += ys_orig[n]**2
    return np.divide(num,den)

class PlotLossesSame:
    def __init__(self, start_epoch, **kwargs):
        self.epochs = [int(start_epoch)]
        self.metrics = list(kwargs.keys())
        plt.ion()

        self.fig, self.axs = plt.subplots(figsize=(8, 5))
        # self.axs.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        for i, (metric, values) in enumerate(kwargs.items()):
            self.__dict__[metric] = [values]
            self.axs.plot([], [], label=metric, alpha=0.7)
        self.axs.grid()
        self.axs.legend()

    def on_epoch_end(self, **kwargs):
        if list(kwargs.keys()) != self.metrics:
            raise ValueError('Need to pass the same arguments as were initialised')

        self.epochs.append(self.epochs[-1] + 1)

        for i, metric in enumerate(self.metrics):
            self.__dict__[metric].append(kwargs[metric])
            self.axs.lines[i].set_data(self.epochs, self.__dict__[metric])
        self.axs.relim()  # recompute the data limits
        self.axs.autoscale_view()  # automatic axis scaling
        self.fig.canvas.flush_events()


class PlotLossesSubPlots:
    def __init__(self, start_epoch, **kwargs):
        self.epochs = [int(start_epoch)]
        inputs = copy.deepcopy(kwargs)
        self.results = inputs

        plt.ion()

        if len(self.results) > 3:
            self.fig, self.axs = plt.subplots(2, (len(self.results) + 1)//2, figsize=(13, 7))
        else:
            self.fig, self.axs = plt.subplots(1, len(self.results), figsize=(13, 5))
        self.axs = self.axs.flatten()

        if isinstance(inputs[list(inputs.keys())[0]], dict):
            self.multlines = True
            for j, (group, results) in enumerate(self.results.items()):
                self.axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.8f'))
                for i, (metric, sub_results) in enumerate(results.items()):
                    if not isinstance(sub_results, list):
                        self.results[group][metric] = [sub_results]
                    self.axs[j].plot(np.arange(len(self.results[group][metric])), self.results[group][metric],
                                     label=metric, alpha=0.5)
                self.axs[j].set_title(group)
                self.axs[j].legend()
                self.axs[j].grid()
        else:
            self.multlines = False
            for j, (metric, values) in enumerate(self.results.items()):
                if not isinstance(values, list):
                    self.results[metric] = [values]
                self.axs[j].plot([], [])
                self.axs[j].set_title(metric)
                self.axs[j].grid()
                self.axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    def on_epoch_end(self, **kwargs):
        # self.epochs.append(self.epochs[-1] + 1)
        inputs = copy.deepcopy(kwargs)

        if self.multlines:
            for i, (group, results) in enumerate(self.results.items()):
                for j, (metric, sub_results) in enumerate(results.items()):
                    if not isinstance(inputs[group][metric], list):
                        sub_results.append(inputs[group][metric])
                    else:
                        sub_results += inputs[group][metric]
                    self.axs[i].lines[j].set_data(np.arange(len(self.results[group][metric])), self.results[group][metric])
                self.axs[i].relim()
                self.axs[i].autoscale_view()
        else:
            for i, (metric, sub_results) in enumerate(self.results.items()):
                sub_results.append(inputs[metric])
                self.axs[i].lines[0].set_data(np.arange(sub_results), sub_results)
                self.axs[i].relim()
                self.axs[i].autoscale_view()

        self.fig.canvas.flush_events()
