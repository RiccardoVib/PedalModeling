import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from GetData2 import get_data
from TrainFunctionality import coefficient_of_determination
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import time

def trainDense(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [4])
    if units[-1] != units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    model_save_dir = kwargs.get('model_save_dir', '../../DenseFeed_TrainedModels')
    save_folder = kwargs.get('save_folder', 'DenseFeed_TESTING')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mae')
    shuffle_data = kwargs.get('shuffle_data', False)
    w_length = kwargs.get('w_length', 0.001)
    n_record = kwargs.get('n_record', 1)

    if data is None:
        #x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=n_record, shuffle=shuffle_data, w_length=w_length, freq='', seed=seed)
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=n_record,
                                                                              shuffle=shuffle_data, w_length=w_length, seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = data

    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit)+', '

    n_units = n_units[:-2]

    #T past values used to predict the next value
    T = x.shape[1] #time window
    D = x.shape[2] #conditioning

    inputs = Input(shape=(T,D), name='input')
    first_unit = units.pop(0)
    if len(units) > 0:
        last_unit = units.pop()
        outputs = Dense(first_unit, name='Dense_0')(inputs)
        for i, unit in enumerate(units):
            outputs = Dense(unit, name='Dense_' + str(i + 1))(outputs)
        outputs = Dense(last_unit, activation='tanh', name='Dense_Fin')(outputs)
    else:
        outputs = Dense(first_unit, activation='tanh', name='Dense')(inputs)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    final_outputs = Dense(1, name='DenseLay')(outputs)
    model = Model(inputs, final_outputs)
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


    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=20, restore_best_weights=True,                                                             verbose=0)
    callbacks += [early_stopping_callback]

    #train
    results = model.fit(x, y, batch_size=b_size, epochs=epochs,
                        validation_data=(x_val, y_val), callbacks=callbacks, verbose=0)

    # #prediction test
    # predictions = []
    # #last train input
    # last_x = x_test[:, :-1]  # DxT array of length T
    #
    # while len(predictions) < len(y_test):
    #     p = model.predict([last_x[0, :], y_test[0, :-1]]) # 1x1 array -> scalar
    #     predictions.append(p)
    #     last_x = np.roll(last_x, -1)
    #
    #     for i in range(last_x.shape[0]):
    #         last_x[-1, i] = p
    #
    #
    # plt.plot(y_test, label='forecast target')
    # plt.plot(predictions, label='forecast prediction')
    # plt.legend()
    predictions_test = model.predict([x_test], batch_size=b_size)

    y_s = np.reshape(y_test, (-1))
    y_pred = np.reshape(predictions_test,(-1))
    r_squared = coefficient_of_determination(y_s[:1600], y_pred[:1600])

    final_model_test_loss = model.evaluate([x_test], y_test, batch_size=b_size, verbose=0)
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate([x_test], y_test, batch_size=b_size, verbose=0)
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
            'drop' : drop,
            'opt_type' : opt_type,
            'loss_type' : loss_type,
            'shuffle_data' : shuffle_data,
            'layers' : layers,
            'n_units' : n_units,
            'n_record' : n_record,
            #'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss'],
            'r_squared': r_squared
        }
        print(results)

    if generate_wav is not None:
        np.random.seed(seed)
        x_gen = x_test
        y_gen = y_test

        #start = time.time()
        predictions = model.predict(x_gen)
        #end = time.time()
        #print(end - start)#0.5176839828491211

        print('GenerateWavLoss: ', model.evaluate([x_gen], y_gen, batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_gen[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_gen)

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'Dense_pred.wav'
        inp_name = 'Dense_inp.wav'
        tar_name = 'Dense_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, 16000, predictions)
        wavfile.write(inp_dir, 16000, x_gen)
        wavfile.write(tar_dir, 16000, y_gen)


    return results

if __name__ == '__main__':
    data_dir = '../Files'
    seed = 422

    trainDense(data_dir=data_dir,
               model_save_dir='../../TrainedModels',
               save_folder='DenseFeed_Testing_prova_input',
               ckpt_flag=False,
               b_size=128,
               learning_rate=0.0001,
               units=[1],
               epochs=1,
               n_record=2,
               loss_type='mse',
               generate_wav=2,
               #w_length=0.00025,
               w_length=0.1,
               shuffle_data=False)