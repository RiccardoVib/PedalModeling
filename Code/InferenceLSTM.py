import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from TrainFunctionality import coefficient_of_determination
from GetData import get_data
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pickle



#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


# generate target given source sequence
def predict_sequence(encoder_model, decoder_model, input_seq, n_steps, output_dim, last_pred, window):
    # encode
    input_seq = input_seq.reshape(1, window, 3)
    state = encoder_model.predict(input_seq)
    # start of sequence input
    target_seq = np.zeros((1, output_dim, 1))  # .reshape(1, 1, output_dim)
    last_prediction = last_pred
    target_seq[0, 0, 0] = last_prediction
    # collect predictions
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = decoder_model.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
        last_prediction = yhat[0, 0, :]

    output = np.array(output)
    return output, last_prediction


def inferenceLSTM(data_dir, epochs, seed=422, data=None, **kwargs):
    
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
    loss_type = kwargs.get('loss_type', 'mse')
    shuffle_data = kwargs.get('shuffle_data', False)
    w_length = kwargs.get('w_length', 16)
    n_record = kwargs.get('n_record', 1)
    inference = kwargs.get('inference', False)

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
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, n_record=n_record, shuffle=shuffle_data, w_length=w_length, seed=seed)
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
    unit_encoder = 8
    unit_decoder = 8
    num_decoder_tokens = 1
    #TRAINING
    
    #encoder
    encoder_inputs = Input(shape=(T,D), name='enc_input')

    outputs, state_h, state_c = LSTM(unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]   
    
    #decoder
    decoder_inputs = Input(shape=(T-1,1), name='dec_input')

    decoder_lstm = LSTM(unit_decoder, return_sequences=True, return_state=True, name='LSTM_De', dropout=drop)
    outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_dense = Dense(num_decoder_tokens, activation='sigmoid', name='DenseLay')
    decoder_outputs = decoder_dense(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    
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

    save_model_best = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_2/Checkpoints'
    best = tf.train.latest_checkpoint(save_model_best)
    model.load_weights(best)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=20, restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]

    #train the RNN
    if not inference:
        results = model.fit([x, y[:, :-1]], y[:, 1:], batch_size=b_size, epochs=epochs, verbose=0,
                            validation_data=([x_val, y_val[:, :-1]], y_val[:, 1:]),
                            callbacks=callbacks)
        # plotting the loss curve over training iteration
        plt.plot(model.loss_curve_)
        plt.xlabel('iteration')
        plt.xlabel('loss')
        plt.show()

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate([x_test, y_test[:, :-1]], y_test[:, 1:], batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    
    #predictions_test = model.predict([x_test, y_test[:, :-1]], batch_size=b_size)

    #final_model_test_loss = model.evaluate([x_test, y_test[:, :-1]], y_test[:, 1:], batch_size=b_size, verbose=0)
    #y_s = np.reshape(y_test[:, 1:], (-1))
    #y_pred = np.reshape(predictions_test,(-1))
    #r_squared = coefficient_of_determination(y_s[:1600], y_pred[:1600])
    
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
            #'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']#,
            #'r_squared': r_squared
        }
        print(results)
    
    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))
        
    
    #INFERENCE
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    #define inference decoder
    decoder_state_input_h = Input(shape=(unit_decoder,))
    decoder_state_input_c = Input(shape=(unit_decoder,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    x_gen = x_test
    y_gen = y_test

    if inference:
        #start = time.time()
        last_prediction = 0
        predictions = []
        output_dim = 1
        for b in range(x_test.shape[0]//4):
            out, last_prediction = predict_sequence(encoder_model, decoder_model, x_test[b, :, :], x_test.shape[1],
                                                    output_dim, last_prediction)
            predictions.append(out)
            #end = time.time()
            #print(end - start)

            x_ = np.zeros((1,2,3))
            x_[0, 0, 0] = x_test[b, 1, 0]
            x_[0, 1, 0] = x_test[b+1, 0, 0]
            x_[0, :, 1] = x_test[b, 0, 1]
            x_[0, :, 2] = x_test[b, 0, 2]
            out, last_prediction = predict_sequence(encoder_model, decoder_model, x_,
                                                    x_test.shape[1],
                                                    output_dim, last_prediction)
            predictions.append(out)
        predictions = np.array(predictions)
    else:
        predictions = model.predict([x_gen, y_gen[:, :-1]])
        print('GenerateWavLoss: ', model.evaluate([x_gen, y_gen[:, :-1]], y_gen[:, 1:], batch_size=b_size, verbose=0))

    predictions = scaler[0].inverse_transform(predictions)
    x_gen = scaler[0].inverse_transform(x_gen[:, :, 0])
    y_gen = scaler[0].inverse_transform(y_gen[:, 1:])

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
    #start = time.time()
    inferenceLSTM(data_dir=data_dir,
              data=data,
              model_save_dir='/Users/riccardosimionato/PycharmProjects/All_Results/Giusti',
              save_folder='LSTM_enc_dec_2_copy',
              ckpt_flag=True,
              b_size=128,
              learning_rate=0.0001,
              encoder_units=[8],
              decoder_units=[8],
              epochs=1,
              loss_type='mse',
              generate_wav=2,
              n_record=27,
              w_length=2,
              shuffle_data=False,
              inference=True)
    #end = time.time()
    #print(end - start)