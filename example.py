import yaml
import neuron.network as N

from data.mnist import load_dataset
import theano.tensor as T


# load training config
config_file = 'data/mnist/mnist_config.yaml'
with open(config_file, 'r') as f:
    network_config = yaml.load(f)

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')


# Build network
layers = network_config["Layers"]
network = N.build_network(layers, input_var)
train_fn, val_fn = N.compile_network(network, input_var, target_var)


# Load the dataset
print("Loading data...")
data = load_dataset()
N.train(data, train_fn, val_fn, num_epochs = 10)







