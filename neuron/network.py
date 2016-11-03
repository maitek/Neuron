import lasagne
import numpy as np
import yaml
import lasagne
import csv
import os
import time

import theano
import theano.tensor as T


"""
    Build Neural Network from list of layers
"""
def build_network(layers, input_var=None):

    # Construct input layer
    input_layer = layers[0]
    input_shape = input_layer['shape']
    network = lasagne.layers.InputLayer(shape=(None, input_shape[2], input_shape[1], input_shape[0]),
                                    input_var=input_var)

    print(layers[0])

    for layer in layers[1:]:
        # Dynamically construct layers from dictionary
        layer_obj = getattr(lasagne.layers, layer['layer'])

        # get layer attributes
        attributes = layer.copy()
        attributes.pop('layer', None)

        # replace string with nonlinearity object
        if "nonlinearity" in attributes.keys():
            attributes["nonlinearity"] = getattr(lasagne.nonlinearities, attributes["nonlinearity"])

        # build layer with attributes
        network = layer_obj(network, **attributes)
        lasagne.layers.get_output(network)
        print(lasagne.layers.get_output_shape(network),layer)
    return network

def compile_network(network, input_var, target_var):  

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn

"""
    Lasagne example, minibatch generator
"""

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train(data, train_fn, val_fn, num_epochs = 10):
    # Unpack data
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Finally, launch the training loop.
    print("Starting training...")

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

"""
    Load data and labels for CSV file


    TODO support string labels by generating a lookup (l_string)
    TODO what to do if input images are of different size
    TODO images could be 1 or 3 channel

"""


def load_csv_data(csv_file, csv_format):
    print("Loading data:", csv_file)
    tic = time.time()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        root, base = os.path.split(csv_file)

        # read labels
        df = pd.read_csv(csv_file, delimiter=',')
        df.columns = csv_format
        # csv_format = ["l_image","l_test","d_test","l_int","loss","l_int"]

        # load labels
        labels_cols = [idx for idx, x in enumerate(csv_format) if x in [
                                                   'l_int', 'l_float', ]]
        y = df.ix[:, labels_cols].as_matrix().astype(np.int32)

        # load images
        if "d_image" in csv_format:
            image_cols = csv_format.index('d_image')
            image_paths = df.ix[:, image_cols]

            X = None
            num_items = len(image_paths.index)
            for idx, row in enumerate(image_paths.items()):
                image = cv2.imread(os.path.join(root, row[1]), 0)  # read grayscale
                image = image.astype(np.float32) / 256

                if len(image.shape) == 2:
                    w, h = image.shape
                    d = 1
                else:
                    w, h, d = image.shape

                if X is None:
                    X = np.zeros((num_items, d, w, h))
                X[idx, :, :, :] = np.swapaxes(image.reshape(1, h, w, d), 1, 3)
            print(X.shape)
        # print(time.time()-tic)

    return X, y
