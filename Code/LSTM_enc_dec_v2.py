#import tensorboard
#load_ext tensorboard
#rm -rf ./logs/
import datetime
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from TrainFunctionality import root_mean_squared_error
from GetData2 import get_data
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pickle


def trainLSTM(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [8, 8])
    decoder_units = kwargs.get('decoder_units', [8, 8])
    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_enc_dec_Testing')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mae')
    shuffle_data = kwargs.get('shuffle_data', False)
    w_length = kwargs.get('w_length', 16)
    n_record = kwargs.get('n_record', 1)

    layers_enc = len(encoder_units)
    layers_dec = len(decoder_units)
    n_units_enc = ''
    for unit in encoder_units:
        n_units_enc += str(unit) + ', '

    n_units_dec = ''
    for unit in decoder_units:
        n_units_dec += str(unit) + ', '

    n_units_enc = n_units_enc[:-2]
    n_units_dec = n_units_dec[:-2]


    if data is None:
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, w_length=w_length, seed=seed)
    else:
        x = data['x']
        y = data['y']
        x_val = data['x_val']
        y_val = data['y_val']
        x_test = data['x_test']
        y_test = data['y_test']
        scaler = data['scaler']
        zero_value = data['zero_value']
        
    #T past values used to predict the next value
    T = x.shape[1] #time window
    D = x.shape[2] #features

    encoder_inputs = Input(shape=(T-1,D), name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(1,1), name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De0', dropout=drop)(decoder_inputs,
                                                                                   initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1), dropout=drop)(outputs)
        outputs, _, _ = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin', dropout=drop)(outputs)
    else:
        outputs, _, _ = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De', dropout=drop)(
                                                                                        decoder_inputs,
                                                                                        initial_state=encoder_states)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_outputs = Dense(1, activation='sigmoid', name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
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
    elif loss_type == 'rmse':
        model.compile(loss=root_mean_squared_error, metrics=['root_mean_squared_error'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    # TODO: Currently not loading weights as we only save the best model... Should probably
    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))
        
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss', mode='min',
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
            
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=20, restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]

    #train
    if not inference:
        results = model.fit([x[:,:-1,:], x[:,-1, 0]], y[:, -1], batch_size=b_size, epochs=epochs, verbose=0,
                            validation_data=([x_val[:,:-1,:], x_val[:,-1, 0]], y_val[:, -1]),
                            callbacks=callbacks)

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
    predictions_test = model.predict([x_test[:, :-1, :], x_test[:, -1, 0].reshape(-1,1)], batch_size=b_size)

    final_model_test_loss = model.evaluate([x_test[:,:-1,:], x_test[:,-1, 0].reshape(-1,1)], y_test[:, -1], batch_size=b_size, verbose=0)
    #y_s = np.reshape(y_test[:, 1:], (-1))
    #y_pred = np.reshape(predictions_test,(-1))
    #r_squared = coefficient_of_determination(y_s[:1600], y_pred[:1600])
    #r2_ = r2_score(y_s[:1600], y_pred[:1600])
    
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate([x_test[:,:-1,:], x_test[:, -1,  0].reshape(-1,1)], y_test[:, -1], batch_size=b_size, verbose=0)
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
            'shuffle_data': shuffle_data,
            'layers_enc': layers_enc,
            'layers_dec': layers_dec,
            'n_units_enc': n_units_enc,
            'n_units_dec': n_units_dec,
            'n_record': n_record,
            'w_length': w_length,
            # 'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss'],
            #'r_squared': r_squared
        }
        print(results)
    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))
        
    if inference:
        n_past = T - 1
        predictions = []
        x_ = x_test.reshape(1, -1, 3)
        for b in range(x_.shape[1] - 2 * T):
            x_i = np.array(x_[:, b:b + n_past, :])
            x_i = x_i.reshape(1,7,3)
            start = time.time()
            out = model.predict([x_i, x_[:, b + T, 0].reshape(1,1,1)])
            predictions.append(out)
            end = time.time()
            print(end - start)

        predictions = np.array(predictions)
        predictions = scaler[0].inverse_transform(predictions)
        predictions = predictions.reshape(-1)
        x_gen = x_test
        y_gen = y_test
        x_gen = scaler[0].inverse_transform(x_gen[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_gen)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        pred_name = 'predictions.wav'
        inp_name = 'inp.wav'
        tar_name = 'tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, 48000, predictions)
        wavfile.write(inp_dir, 48000, x_gen)
        wavfile.write(tar_dir, 48000, y_gen)

    if not inference:
        np.random.seed(seed)
        x_gen = x_test
        y_gen = y_test
        predictions = model.predict([x_gen[:, :-1, :], x_gen[:, -1, 0].reshape(-1,1)])
        print('GenerateWavLoss: ', model.evaluate([x_gen[:, :-1, :], x_gen[:, -1, 0].reshape(-1,1)], y_gen[:, -1], batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_gen[:, -1, 0])
        y_gen = scaler[0].inverse_transform(y_gen[:, -1])

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

            # Save some Spectral Plots:
    #         spectral_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'SpectralPlots'))
    #         if not os.path.exists(spectral_dir):
    #             os.makedirs(spectral_dir)
    #         plot_spectral(Zxx=predictions[i], title='Predictions',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, pred_name)).replace('.wav', '.png'))
    #         plot_spectral(Zxx=x_gen[i], title='Inputs',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, inp_name)).replace('.wav', '.png'))
    #         plot_spectral(Zxx=y_gen[i], title='Target',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, tar_name)).replace('.wav', '.png'))
    #
    #
    return results


if __name__ == '__main__':
    data_dir = '../Files'
    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w8.pickle'])), 'rb')
    data = pickle.load(file_data)
    
    seed = 422
    #start = time.time()
    trainLSTM(data_dir=data_dir,
              data=data,
              model_save_dir='/Users/riccardosimionato/PycharmProjects/All_Results/Giusti',
              save_folder='LSTM_enc_dec_v2_8',
              ckpt_flag=True,
              b_size=128,
              learning_rate=0.0001,
              encoder_units=[8],
              decoder_units=[8],
              epochs=100,
              loss_type='mse',
              generate_wav=None,
              n_record=27,
              w_length=16,
              shuffle_data=False,
              inference=True)
    #end = time.time()
    #print(end - start)