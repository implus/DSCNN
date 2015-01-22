from __future__ import print_function

import itertools
import time
import lasagne
import theano
import theano.tensor as T

from lasagne.layers import cuda_convnet

import numpy as np
import cPickle as pickle


NUM_EPOCHS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

def _load_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data

def load_data():
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
 #   X_train, y_train = data[0]
 #   X_valid, y_valid = data[1]
 #   X_test, y_test = data[2]

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 3, 32, 32)) 
    y_train = np.array(y_train, dtype = 'int32') 
    X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))
    y_test  = np.array(y_test, dtype = 'int32')
    X_valid = X_test
    y_valid = y_test

    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_width=X_train.shape[2],
        input_height=X_train.shape[3],
        output_dim=10,
        )


def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE, dimshuffle=True):
    l_in = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, 3, input_width, input_height),
        )

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=64,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )
    l_pool1 = cuda_convnet.MaxPool2DCCLayer(
        l_conv1,
        ds=(3, 3),
        strides=(2, 2),
        dimshuffle=dimshuffle,
        )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=64,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )
    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        ds=(3, 3),
        strides=(2, 2),
        dimshuffle=dimshuffle,
        )

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool2,
        num_filters=64,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )
    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv3,
        ds=(3, 3),
        strides=(2, 2),
        dimshuffle=dimshuffle,
        )

    l_conv4 = cuda_convnet.Conv2DCCLayer(
        l_pool3,
        num_filters=64,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )
    l_pool4 = cuda_convnet.MaxPool2DCCLayer(
        l_conv4,
        ds=(3, 3),
        strides=(2, 2),
        dimshuffle=dimshuffle,
        )
    
    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool4,
        num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units = 128,
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


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    def loss(output):
        return -T.mean(T.log(output)[T.arange(y_batch.shape[0]), y_batch])

    loss_train = loss(output_layer.get_output(X_batch))
    loss_eval = loss(output_layer.get_output(X_batch, deterministic=True))

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
            },
        )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            },
        )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            },
        )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    #num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
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

def main(num_epochs=NUM_EPOCHS):
    dataset = load_data()

    output_layer = build_model(
        input_width=dataset['input_width'],
        input_height=dataset['input_width'],
        output_dim=dataset['output_dim'],
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor4,
        )

    print("Starting training...")
    last = time.clock()
    mx = 0.0
    mxtime = 0
    lastfn = ''
    import os
    for epoch in train(iter_funcs, dataset):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
            break
        now = epoch['valid_accuracy'] * 100
        fn = 'model_64cp_64cp_64cp_64cp_128_128'
        if now > mx:
            mx = now
            mxtime = epoch['number']

            os.system('rm -f ' + lastfn)
            tmpfn = fn + '_accuracy_' + str(mx) + '_epoch_' + str(mxtime)
            with open(tmpfn, 'wb') as f:
                pickle.dump(output_layer, f, -1)
            lastfn = tmpfn

        print("time use " + str(time.clock() - last) + "s")
        print(fn)
        print("max accuracy till now : " + str(mx) + "% first appear time is " + str(mxtime))
        last = time.clock()

    return output_layer

import sys
sys.setrecursionlimit(1000000)

if __name__ == '__main__':
    model = main()
    fn = 'model_64cp_64cp_64cp_64cp_128_128'
    with open(fn, 'wb') as f:
        pickle.dump(model, f, -1)
