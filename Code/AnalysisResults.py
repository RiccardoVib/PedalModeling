import os
import pickle
import matplotlib.pyplot as plt

data_dir = '../Files'

file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16_[-1,1].pickle'])), 'rb')
data = pickle.load(file_data)

x = data['x']
y = data['y']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']
scaler = data['scaler']
zero_value = data['zero_value']
fs = int(data['fs'])

fig, ax = plt.subplots()
ax.plot(x[10:10000 ,:,0].reshape(-1))
