from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow.keras as tk
import tensorflow as tf

import nengo_dl
from kerasModel import kmodel
from utils import getData
import random
import nni
import os

def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  #os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  rng = np.random.RandomState(seed)
  print(f"Random seed set as {seed}")


def train_nengo_cnn(converter, inputs, train, validation):

    inp, dense = inputs
    x_train, y_train = train
    x_val, y_val = validation

    with nengo_dl.Simulator(converter.net, minibatch_size=404) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[inp]: x_train},
            {converter.outputs[dense]: y_train},
            validation_data=(
                {converter.inputs[inp]: x_val},
                {converter.outputs[dense]: y_val},
            ),
            epochs=2,
        )

        # save the parameters to file
        sim.save_params("./keras_to_snn_params")

def run_network(
    model,
    inputs,
    test,
    activation,
    params_file="keras_to_snn_params",
    n_steps=30,
    scale_firing_rates=1,
    synapse=None
):
    inp, conv1, dense = inputs
    x_test, y_test = test
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        swap_activations={tk.activations.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[dense]

    # add a probe to the first convolutional layer to record activity.
    # we'll only record from a subset of neurons, to save memory.
    sample_neurons = np.linspace(
        0,
        np.prod(conv1.shape[1:]),
        1000,
        endpoint=False,
        dtype=np.int32,
    )
    with nengo_converter.net:
        conv1_probe = nengo.Probe(nengo_converter.layers[conv1][sample_neurons])

    # repeat inputs for some number of timesteps
    tiled_test_images = np.tile(x_test[:1240], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=10, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})

    # compute accuracy on test data, using output of network on
    # last timestep
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)

    accuracy = (predictions == y_test[:1240, 0, 0]).mean()
    print(f"Test accuracy: {100 * accuracy:.2f}%")

    nni.report_final_result(accuracy)

    plot_1 = False
    if plot_1:
      for ii in range(3):
          plt.figure(figsize=(12, 4))

          plt.subplot(1, 3, 1)
          plt.title(f"y test: {y_test[ii]}")
          plt.imshow(x_test[ii, 0].reshape((33, 57)))
          plt.axis("off")

          plt.subplot(1, 3, 2)
          scaled_data = data[conv1_probe][ii] * scale_firing_rates
          if isinstance(activation, nengo.SpikingRectifiedLinear):
              scaled_data *= 0.001
              rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
              plt.ylabel("Number of spikes")
          else:
              rates = scaled_data
              plt.ylabel("Firing rates (Hz)")
          plt.xlabel("Timestep")
          plt.title(
              f"Neural activities (conv0 mean={rates.mean():.1f} Hz, "
              f"max={rates.max():.1f} Hz), "
              f"sfr= {scale_firing_rates}"
          )
          plt.plot(scaled_data)

          plt.subplot(1, 3, 3)
          plt.title("Output predictions")
          plt.plot(tf.nn.softmax(data[nengo_output][ii]))
          plt.legend([str(j) for j in range(10)], loc="upper left")
          plt.xlabel("Timestep")
          plt.ylabel("Probability")

          plt.tight_layout()

          plt.savefig(f'{y_test[ii]}.jpg')
            
def main():
    set_seed()
      
    x_train, y_train, x_val, y_val, x_test, y_test = getData()

    print(f'train oh: {y_train[0]}')

    inp, conv0, conv1, dense, model = kmodel([x_train, y_train], [x_val, y_val], [x_test, y_test])

    y_train = np.argmax(y_train, axis=-1)
    y_val = np.argmax(y_val, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    x_train = x_train.reshape((x_train.shape[0], 1, -1))
    y_train = y_train.reshape((y_train.shape[0], 1, -1))
    x_val = x_val.reshape((x_val.shape[0], 1, -1))
    y_val = y_val.reshape((y_val.shape[0], 1, -1))
    x_test = x_test.reshape((x_test.shape[0], 1, -1))
    y_test = y_test.reshape((y_test.shape[0], 1, -1))

    print(f'train and train labels: {x_train.shape} and {y_train.shape}')
    print(f'val and val labels: {x_val.shape} and {y_val.shape}')

    converter = nengo_dl.Converter(model)

    do_training = False

    if do_training:
      train_nengo_cnn(converter, [inp, dense], [x_train, y_train], [x_val, y_val])
    else:
      urlretrieve(
        "https://drive.google.com/uc?export=download&"
        "id=1eDiwlxb5xmehVOpK18k1mXg5ceJQuXbb",	
        "keras_to_snn_params.npz",
        )

    # activations = {
    #     1: nengo.SpikingRectifiedLinear(),
    #     2: nengo.LIF(),
    #     3: nengo.AdaptiveLIF(),
    #     4: nengo.RegularSpiking(),
    #     5: nengo.PoissonSpiking(),
    #     6: nengo.StochasticSpiking()
    # }

    #run_network(model, [inp, conv0, dense], [x_test, y_test], activation=nengo.RectifiedLinear(), n_steps=10)
    
    activations = {
        1: nengo.SpikingRectifiedLinear(),
        2: nengo.LIF(),
        3: nengo.AdaptiveLIF()
    }

    params = {
        's': 0.01,
        'n_steps': 90,
        'activation': 1,
        'scale_firing_rates':300
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
      
    run_network(
        model, [inp, conv1, dense], [x_test, y_test],
        activation=activations[params['activation']],
        n_steps=params['n_steps'],
        synapse=params['s'],
        scale_firing_rates = params['scale_firing_rates']
    )


if __name__=="__main__":
    main()