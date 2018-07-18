import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.data_utils import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

def construct_customnet (x,y):

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

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


def train_customnet (data, epochs, validation_size, x, y):

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

def train_with_eval_customnet (data, eval_datasets):

    directory_path = "/home/jrnf24/data/"

    # Load hdf5 dataset
    h5f = h5py.File('{}{}.h5'.format(directory_path,data), 'r')
    X = h5f['X']
    Y = h5f['Y']

    model = construct_customnet(224,224)

    model.fit(X, Y, n_epoch=5, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("5", eval_datasets, model)

    model.fit(X, Y, n_epoch=5, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("10", eval_datasets, model)

    model.fit(X, Y, n_epoch=10, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("20", eval_datasets, model)

    model.fit(X, Y, n_epoch=10, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("30", eval_datasets, model)

    model.fit(X, Y, n_epoch=20, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("50", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("75", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='customnet')

    evaluate_in_train("100", eval_datasets, model)

    h5f.close()
