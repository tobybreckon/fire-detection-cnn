import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

def construct_customnet (x,y):

    network = tflearn.input_data(shape=[None, y, x, dims], dtype=tf.float32)

    network = conv_2d(network, 64, 3, strides=4, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Training
    model = tflearn.DNN(network, checkpoint_path='model_customnet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model


def train_customnet (data, epochs, validation_size, x, y)

directory_path = "/home/jrnf24/data/"

    # Load hdf5 dataset
    h5f = h5py.File('{}{}.h5'.format(directory_path,data), 'r')
    X = h5f['X']
    Y = h5f['Y']

    model = construct_customnet(x,y)

    # Training
    model.fit(X, Y, n_epoch=epochs, validation_set=validation_size, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    h5f.close()
