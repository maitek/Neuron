import theano
import theano.tensor as T
import numpy as np
import lasagne


def ascii_encode_string(input_string):
    code = np.fromstring(input_string,dtype=np.uint8)
    return code

def ascii_decode_string(input_code):
    string = ''.join([chr(c) for c in input_code])
    return string


# ############################# Batch iterators###############################

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


def iterate_minibatches_text(file_object, code_len = 64, batch_size = 128, shuffle = False):
    """
        Generator that lazy-reads a file object and yields a
        mini-batch of encoded strings as X and next encoded character as y
    """
    # Allocate batch
    X_batch = np.empty((batch_size,code_len), dtype=np.float32)
    y_batch = np.empty(batch_size, dtype=np.int32)

    while True:

        # read batch chunk
        data = file_object.read(code_len+batch_size)
        if not data or len(data) < code_len+batch_size:
            break
        code = ascii_encode_string(data)

        # create batch by sliding a window over code
        for i in range(batch_size):
            X_batch[i,:] = code[i:i+code_len]
            y_batch[i] = int(code[i+code_len])

        yield X_batch, y_batch


def classification(network, input_var):

    target_var = T.ivector('targets')

    # Setup theano functions for classification
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

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    pred_fn  = theano.function([input_var], [test_prediction])

    return train_fn, val_fn, pred_fn
