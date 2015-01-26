from __future__ import print_function

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


NUM_EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
LAMDA = 1.0 # adjust new feature's value weight

FEATURE_LAYER = 'FEATURE_LAYER_'

def _load_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data

def load_data(depth = 0):

    if depth == 0:
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
        y_train = np.array(y_train, dtype = 'int32') 
        X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))
        y_test  = np.array(y_test, dtype = 'int32')
        X_valid = X_test
        y_valid = y_test

    else:
        #fn = FEATURE_LAYER + str(depth - 1)
        #data = _load_data(fn)
        data = np.load(outfile)
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']

    print("X_train shape", X_train.shape)
    print("X_valid shape", X_valid.shape)

    return dict(
        X_train_value = np.cast["float32"](X_train),
        X_valid_value = np.cast["float32"](X_valid),
        X_train=theano.shared(lasagne.utils.floatX(X_train), borrow = True),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid), borrow = True),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        input_width=X_train.shape[2],
        input_height=X_train.shape[3],
        input_dim = X_train.shape[1],
        output_dim=10,
        )


def build_model(input_width, input_height, input_dim, output_dim, dep = 0,
                batch_size=BATCH_SIZE, dimshuffle=True):
    l_in = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, input_dim, input_width, input_height),
        )

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=96,
        pad = 1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        dimshuffle=dimshuffle,
        )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters= 32 * (2**dep),
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


def create_iter_functions(dataset, model,
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

    loss_train = loss(model.get_output(X_batch))
    loss_eval = loss(model.get_output(X_batch, deterministic=True))

    pred = T.argmax(
        model.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(model)
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

    return dict(
        train=iter_train,
        valid=iter_valid,
        )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144] )

def gao(depth, num_epochs=NUM_EPOCHS, lamda = LAMDA):
    dataset = load_data(depth)

    model = build_model(
        input_width=dataset['input_width'],
        input_height=dataset['input_width'],
        input_dim = dataset['input_dim'],
        output_dim=dataset['output_dim'],
        dep = depth,
        )

    iter_funcs = create_iter_functions(
        dataset,
        model,
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

        now = epoch['valid_accuracy'] * 100
        fn = 'model_64cp_32cp_128_128'
        if now > mx:
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

    #X_train = X_train / 255.0
    #X_test  = X_test  / 255.0

    X_train = X_train.reshape((X_train.shape[0], 3, 32, 32)) 
    y_train = np.array(y_train, dtype = 'int32') 
    X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))
    y_test  = np.array(y_test, dtype = 'int32')
    X_valid = X_test
    y_valid = y_test

# latest pool layer
    model = _load_data(lastfn)
    labels_train = y_train
    labels_valid = y_valid

    input = T.tensor4('input')
    # change if model structure has been changed
    expre = model.input_layer.input_layer.input_layer.input_layer.input_layer.get_output(input)

    batch_size = BATCH_SIZE
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    
    t = expre.eval({input: dataset['X_train_value'][0: batch_size]})
    data_train = np.zeros((dataset['num_examples_train'], t.shape[1], t.shape[2], t.shape[3]))
    for i in range(0, num_batches_train):
        batch_slice = slice(
            i * batch_size, (i + 1) * batch_size)
        tmp = expre.eval({input: dataset['X_train_value'][batch_slice]})
        data_train[batch_slice] = tmp

    t = expre.eval({input: dataset['X_valid_value'][0: batch_size]})
    data_valid = np.zeros((dataset['num_examples_valid'], t.shape[1], t.shape[2], t.shape[3]))
    for i in range(0, num_batches_valid):
        batch_slice = slice(
            i * batch_size, (i + 1) * batch_size)
        tmp = expre.eval({input: dataset['X_valid_value'][batch_slice]})
        data_valid[batch_slice] = tmp

    print("data_train shape", data_train.shape, "data_valid shape", data_valid.shape)
    
    #randomw = random.randint(0, data_train.shape[0])
    nh = data_train.shape[2]
    nw = data_train.shape[3]
    base_data_train = np.zeros((X_train.shape[0], 4, nh, nw))
    base_data_valid = np.zeros((X_valid.shape[0], 4, nh, nw))
    for i in range(0, X_train.shape[0]):
        t = X_train[i]
        t = t.swapaxes(0, 2).swapaxes(0, 1) # from (3, h, w) -> (h, w, 3)
        img = Image.fromarray(t, mode = "RGB")
        nimg = img.resize((nh, nw), Image.NEAREST)
        #if i == randomw:
        #    imgplot = plt.imshow(nimg)
        #    plt.show(imgplot)
        arr = np.array(nimg) # (nh, nw, 3)
        gray = rgb2gray(arr).reshape(nh, nw, 1)
        arr = np.concatenate((arr, gray), axis = 2)
        arr = arr.swapaxes(0, 2).swapaxes(1, 2)
        base_data_train[i] = arr / 255.0

    for i in range(0, X_valid.shape[0]):
        t = X_valid[i]
        t = t.swapaxes(0, 2).swapaxes(0, 1)
        img = Image.fromarray(t, mode = "RGB")
        nimg = img.resize((nh, nw), Image.NEAREST)
        #if i == randomw:
        #    imgplot = plt.imshow(nimg)
        #    plt.show(imgplot)
        arr = np.array(nimg) # (nh, nw, 3)
        gray = rgb2gray(arr).reshape(nh, nw, 1)
        arr = np.concatenate((arr, gray), axis = 2)
        arr = arr.swapaxes(0, 2).swapaxes(1, 2)
        base_data_valid[i] = arr / 255.0

    data_train = np.concatenate((base_data_train, data_train * lamda), axis = 1)
    data_valid = np.concatenate((base_data_valid, data_valid * lamda), axis = 1)
    
    # deal with new files
    # fn = FEATURE_LAYER + str(depth - 1)
    # data = _load_data(fn)
    # X_train = data['X_train']
    # y_train = data['y_train']
    # X_valid = data['X_valid']
    # y_valid = data['y_valid']
    
    # dictionary = dict( X_train =  data_train, y_train =  labels_train, X_valid =  data_valid, y_valid =  labels_valid,)
    fn = FEATURE_LAYER + str(depth)
    print(fn + " file store last result")
    global outfile 
    outfile = TemporaryFile()
    np.savez(outfile, X_train = data_train, y_train = labels_train, X_valid = data_valid ,y_valid = labels_valid)
    outfile.seek(0)


    #with open(fn, 'wb') as f:
    #   pickle.dump(dictionary, f, -1)

    print("gao ok")

if __name__ == '__main__':
    for i in range(0, 4):
        gao(depth = i)
