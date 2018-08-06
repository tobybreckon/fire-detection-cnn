import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.data_utils import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

#directory_path = "/Users/Andrew/Documents/Project/Data"
directory_path = "/Volumes/AJD/furg-fire-dataset/Data/"
rows = 240
cols = 320
dims = 3

# Load hdf5 dataset
h5f = h5py.File('%sdata.h5' % directory_path, 'r')
X = h5f['X']
Y = h5f['Y']
X_test = h5f['X_test']
Y_test = h5f['Y_test']

# Build network
net = tflearn.input_data(shape=[None, cols, rows, dims], dtype=tf.float32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, run_id='cnn')

h5f.close()
