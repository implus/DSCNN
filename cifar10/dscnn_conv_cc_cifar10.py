from __future__ import print_function

import os
import random
from tempfile import TemporaryFile
import itertools
import time
import lasagne
import theano
import theano.tensor as T
from PIL import Image
#import matplotlib.pyplot as plt
#import random
import sys
sys.setrecursionlimit(10000000)

from lasagne.layers import cuda_convnet

import numpy as np
import cPickle as pickle


NUM_EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
LAMDA = 1.0 # adjust new feature's value weight
FILE_MODEL = 'model_256+64*dc_128*2**dcp_1024fp_1024fp_cifar10'

FEATURE_LAYER = 'FEATURE_LAYER_'
FILE_PREFIX = './cifar10_dataset/'

def _load_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data

def build_model(input_width, input_height, input_dim, output_dim, dep = 0,
                batch_size=BATCH_SIZE, dimshuffle=True):
    shape=(BATCH_SIZE, input_dim, input_width, input_height)
    print("build model input: ", shape)
    l_in = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, input_dim, input_width, input_height),
        )

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters= 256 + 64 * dep,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )
    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters= 128 * (2 ** dep),
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )

    if input_width > 3:
        l_pool2 = cuda_convnet.MaxPool2DCCLayer(
            l_conv2,
            ds=(3, 3),
            strides=(2, 2),
            dimshuffle=dimshuffle,
            )
    else:
        l_pool2 = cuda_convnet.MaxPool2DCCLayer(
            l_conv2,
            ds=(2, 2),
            strides=(2, 2),
            dimshuffle=dimshuffle,
            )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units = 1024,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units = 1024,
        nonlinearity = lasagne.nonlinearities.rectify,
        W = lasagne.init.Uniform(),
        )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units = output_dim,
        nonlinearity = lasagne.nonlinearities.softmax,
        )

    return l_out


def create_iter_functions_seperate(model, X_tensor_type = T.tensor4,
                          learning_rate = LEARNING_RATE, momentum = MOMENTUM):
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    
    def loss(output):
        return -T.mean(T.log(output)[T.arange(y_batch.shape[0]), y_batch])

    loss_train = loss(model.get_output(X_batch))
    loss_eval  = loss(model.get_output(X_batch, deterministic = True))

    pred = T.argmax(
        model.get_output(X_batch, deterministic = True), axis = 1)

    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(model)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [X_batch, y_batch], loss_train,
        updates = updates,
        )

    iter_valid = theano.function(
        [X_batch, y_batch], [loss_eval, accuracy],
        )

    return dict(
        train = iter_train,
        valid = iter_valid,
        )

def predeal(data_filename_prefix_train, 
            data_filename_prefix_valid, 
            batch_size = BATCH_SIZE ):
    fn = './cifar-10-batches-py/data_batch_' + str(1)
    print(fn + 'is loading')
    data = _load_data(fn)
    X_train = data['data']
    y_train = data['labels']
    
    for w in range(2, 6):
        fn = './cifar-10-batches-py/data_batch_' + str(w)
        print(fn + 'is loading')
        data = _load_data(fn)
        X_train = np.concatenate((X_train, data['data']))
        y_train = np.concatenate((y_train, data['labels']))


    fn = './cifar-10-batches-py/test_batch'
    print(fn + 'is loading')
    data = _load_data(fn)
    X_test = data['data']
    y_test = data['labels']

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 3, 32, 32)) 
    X_train = np.cast["float32"](X_train)
    y_train = np.array(y_train, dtype = 'int32') 
    X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))
    X_test = np.cast["float32"](X_test)
    y_test  = np.array(y_test, dtype = 'int32')
    X_valid = X_test
    y_valid = y_test

    filenum_train = X_train.shape[0] // batch_size
    filenum_valid = X_valid.shape[0] // batch_size

    print('filenum_train = ', filenum_train, 'filenum_valid = ', filenum_valid)

    for b in range(filenum_train):
        filename_train = data_filename_prefix_train + '_' + str(b) + '.npz'
        batch_slice = slice(
            b * batch_size, (b + 1) * batch_size)
        if b == 0:
            print('first save X_train shape = ', X_train[batch_slice].shape, 
                 'first save y_train shape = ', y_train[batch_slice].shape)
        np.savez(filename_train, data = X_train[batch_slice], labels = y_train[batch_slice])

    for b in range(filenum_valid):
        filename_valid = data_filename_prefix_valid + '_' + str(b) + '.npz'
        batch_slice = slice(
            b * batch_size, (b + 1) * batch_size
            )
        if b == 0:
            print('first save X_valid shape = ', X_valid[batch_slice].shape, 
                 'first save y_valid shape = ', y_valid[batch_slice].shape)
        np.savez(filename_valid, data = X_valid[batch_slice], labels = y_valid[batch_slice])

    return filenum_train, filenum_valid
    # np.savez(filename, X_train = data_train, y_train = labels_train, X_valid = data_valid ,y_valid = labels_valid)



def train(iter_funcs, data_filename_prefix_train, data_filename_prefix_valid, data_filenum_train, data_filenum_valid, depth):
    for epoch in itertools.count(1):
        batch_train_losses = []
        index_array = range(data_filenum_train)
        random.shuffle(index_array)
        for b in index_array:
            # when predeal train, store data as _xx with 'data' and 'labels' labels
            data_train = data_filename_prefix_train + '_' + str(b) + '.npz'
            data = np.load(data_train)
            X_train = np.cast['float32'](data['data'])
            y_train = np.cast['int32'](data['labels'])
            batch_train_loss = iter_funcs['train'](X_train, y_train)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(data_filenum_valid):
            # when predeal test, store data as _xx with 'data' and 'labels' labels
            data_valid = data_filename_prefix_valid + '_' + str(b) + '.npz'
            data = np.load(data_valid)
            X_valid = np.cast['float32'](data['data'])
            y_valid = np.cast['int32'](data['labels'])
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](X_valid, y_valid)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


def load_info(data_filename_prefix_train):
    tmp = np.load(data_filename_prefix_train + '_0.npz')
    X_train = tmp['data']
    return dict(
        input_width=X_train.shape[2],
        input_height=X_train.shape[3],
        input_dim = X_train.shape[1],
        output_dim=10,
        )
    
def savedata_forstack(model,
             data_filename_prefix_train, 
             data_filename_prefix_valid, 
             data_filenum_train, 
             data_filenum_valid ):

    input = T.tensor4('input')
    expre = model.input_layer.input_layer.input_layer.input_layer.input_layer.get_output(input)

    for b in range(data_filenum_train):
        # when predeal train, store data as _xx with 'data' and 'labels' labels
        filename_train = data_filename_prefix_train + '_' + str(b) + '.npz'
        data = np.load(filename_train)
        X_train = np.cast['float32'](data['data']) 
        y_train = data['labels'] 

        # X_train (100, 3, 32, 32) add (100, 256, 16, 16)
        add = expre.eval({input: X_train})
        nh = add.shape[2]
        nw = add.shape[3]
        data_train = np.zeros((X_train.shape[0], X_train.shape[1], nh, nw))
        for w in range(X_train.shape[0]):
            for d in range(X_train.shape[1]):
                img = Image.fromarray(X_train[w][d], mode = 'F') 
                nimg = img.resize((nh, nw), Image.ANTIALIAS)
                arr = np.array(nimg).reshape(nh, nw)
                data_train[w][d] = arr
        data_train = np.concatenate((data_train, add), axis = 1)
        has = data_train.shape[1]
        left = (has + 3) // 4 * 4 - has
        if left > 0:
            data_train = np.concatenate((data_train, np.zeros((X_train.shape[0], left, nh, nw))), axis = 1)
        if b == 0:
            print('save one train file shape: ', data_train.shape)
        np.savez(filename_train, data = data_train, labels = y_train)


    for b in range(data_filenum_valid):
        # when predeal test, store data as _xx with 'data' and 'labels' labels
        filename_valid = data_filename_prefix_valid + '_' + str(b) + '.npz'
        data = np.load(filename_valid)
        X_valid = np.cast['float32'](data['data'])
        y_valid = data['labels']

        add = expre.eval({input: X_valid})
        nh = add.shape[2]
        nw = add.shape[3]
        data_valid = np.zeros((X_valid.shape[0], X_valid.shape[1], nh, nw))
        for w in range(X_valid.shape[0]):
            for d in range(X_valid.shape[1]):
                img = Image.fromarray(X_valid[w][d], mode = 'F')
                nimg = img.resize((nh, nw), Image.ANTIALIAS)
                arr = np.array(nimg).reshape(nh, nw)
                data_valid[w][d] = arr
        data_valid = np.concatenate((data_valid, add), axis = 1)
        has = data_valid.shape[1]
        left = (has + 3) // 4 * 4 - has
        if left > 0:
            data_valid = np.concatenate((data_valid, np.zeros((X_valid.shape[0], left, nh, nw))), axis = 1)
        if b == 0:
            print('save one valid file shape: ', data_valid.shape)
        np.savez(filename_valid, data = data_valid, labels = y_valid)

    # np.savez(filename_train, data = X_train, labels = y_train)

def gao(depth, 
        data_filename_prefix_train, 
        data_filename_prefix_valid,
        data_filenum_train,
        data_filenum_valid,
        num_epochs=NUM_EPOCHS, lamda = LAMDA):

    datainfo = load_info(data_filename_prefix_train)
    print("gao data shape:", datainfo)

    model = build_model(
        input_width=datainfo['input_width'],
        input_height=datainfo['input_width'],
        input_dim = datainfo['input_dim'],
        output_dim=datainfo['output_dim'],
        dep = depth,
        )

    iter_funcs = create_iter_functions_seperate(
        model,
        )


    print("Starting training...")
    last = time.clock()
    mx = 0.0
    mxtime = 0
    lastfn = ''
    for epoch in train(iter_funcs, 
                       data_filename_prefix_train, 
                       data_filename_prefix_valid,
                       data_filenum_train,
                       data_filenum_valid,
                       depth = depth,
                      ):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        now = epoch['valid_accuracy'] * 100
        fn = FILE_MODEL
        if now > mx + 0.000001:
            mx = now
            mxtime = epoch['number']

            os.system('rm -f ' + lastfn)
            tmpfn = fn + '_accuracy_' + str(mx) + '_epoch_' + str(mxtime) + '_depth_' + str(depth)
            with open(tmpfn, 'wb') as f:
                pickle.dump(model, f, -1)
            lastfn = tmpfn

        print("time use " + str(time.clock() - last) + "s")
        print(fn)
        print("Depth = " + str(depth) + " max accuracy till now : " + str(mx) + "% first appear time is " + str(mxtime))
        last = time.clock()
        if epoch['number'] >= num_epochs:
            break

    savedata_forstack(_load_data(lastfn), 
             data_filename_prefix_train, 
             data_filename_prefix_valid, 
             data_filenum_train, 
             data_filenum_valid )
    
    print("gao ok")
    return


if __name__ == '__main__':
    data_filename_prefix_train = FILE_PREFIX + 'train'
    data_filename_prefix_valid = FILE_PREFIX + 'valid'
    # first predeal to get seperate dataset, get data file train num, valid num
    data_filenum_train, data_filenum_valid = predeal(data_filename_prefix_train, data_filename_prefix_valid)

    for i in range(0, 5):
        gao(i, data_filename_prefix_train, data_filename_prefix_valid, data_filenum_train, data_filenum_valid)
