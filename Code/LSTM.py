import tensorboard
#load_ext tensorboard
#rm -rf ./logs/
import datetime
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from TrainFunctionality import coefficient_of_determination
from GetData2 import get_data
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score
import pickle
#
def trainLSTM(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [1])
    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_testing')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mse')
    shuffle_data = kwargs.get('shuffle_data', False)
    w_length = kwargs.get('w_length', 16)
    n_record = kwargs.get('n_record', 1)

    if data is None:
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=n_record,
                                                                              shuffle=shuffle_data, w_length=w_length,
                                                                              seed=seed)
    else:
        x = data['x']
        y = data['y']
        x_val = data['x_val']
        y_val = data['y_val']
        x_test = data['x_test']
        y_test = data['y_test']
        scaler = data['scaler']
        zero_value = data['zero_value']

    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit)+', '

    n_units = n_units[:-2]

    #T past values used to predict the next value
    T = x.shape[1] #time window
    D = x.shape[2]

    inputs = Input(shape=(T,D), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs = LSTM(last_unit_encoder, name='LSTM_EnFin')(outputs)
    else:
        outputs = LSTM(first_unit_encoder, name='LSTM_En')(inputs)

    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    outputs = Dense(1, name='DenseLay')(outputs)
    model = Model(inputs, outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(
            os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                                  mode='min',
                                                                  save_best_only=False, save_weights_only=True,
                                                                  verbose=1)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        latest = tf.train.latest_checkpoint(ckpt_dir_latest)
        if latest is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(latest)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=20, restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]

    #train the RNN
    if not inference:
        results = model.fit(x, y[:, 0], batch_size=b_size, epochs=epochs,  verbose=0,
                        validation_data=(x_val, y_val[:, 0]),
                        callbacks=callbacks)

    predictions_test = model.predict(x_test, batch_size=b_size)

    final_model_test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    #y_s = np.reshape(y_test, (-1))
    #y_pred = np.reshape(predictions_test,(-1))
    #r_squared = coefficient_of_determination(y_s[:1600], y_pred[:1600])
    #r2_ = r2_score(y_s[:1600], y_pred[:1600])

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)

    test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'learning_rate': learning_rate,
            'drop': drop,
            'opt_type': opt_type,
            'loss_type': loss_type,
            'layers': layers,
            'units': n_units,
            'n_record': n_record,
            'w_length': w_length,
            #'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']
        }
        print(results)

    if generate_wav is not None:
        np.random.seed(seed)
        x_gen = x_test
        y_gen = y_test
        predictions = model.predict(x_gen)
        print('GenerateWavLoss: ', model.evaluate(x_gen, y_gen[:, 0], batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_gen[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_gen[:, 0])

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'LSTM_pred.wav'
        inp_name = 'LSTM_inp.wav'
        tar_name = 'LSTM_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, 48000, predictions)
        wavfile.write(inp_dir, 48000, x_gen)
        wavfile.write(tar_dir, 48000, y_gen)

    return results

if __name__ == '__main__':
    data_dir = '../Files'
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w2.pickle'])), 'rb')
    data = pickle.load(file_data)

    seed = 422
    trainLSTM(data_dir=data_dir,
              data=data,
              model_save_dir='../../TrainedModels',
              save_folder='LSTM_Testing',
              ckpt_flag=True,
              b_size=128,
              units=[8, 8],
              learning_rate=0.0001,
              epochs=1,
              loss_type='mse',
              generate_wav=2,
              n_record=27,
              w_length=2,
              shuffle_data=False,
              inference=False)