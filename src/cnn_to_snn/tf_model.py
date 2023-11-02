from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tk
from utils import normalize, extract_label, label_processing, top_10_dataset 
import numpy as np

DEBUG=True

def get_layers(phenotype):
    """
        Parses the phenotype corresponding to the layers.
        Auxiliary function of the assemble_network function.

        Parameters
        ----------
        phenotye : str
            individual layers phenotype

        Returns
        -------
        layers : list
            list of tuples (layer_type : str, node properties : dict)
    """

    raw_phenotype = phenotype.split(' ')
    print(raw_phenotype)
    idx = 0
    first = True
    node_type, node_val = raw_phenotype[idx].split(':')
    layers = []

    while idx < len(raw_phenotype):
        if node_type == 'layer':
            if not first:
                layers.append((layer_type, node_properties))
            else:
                first = False
            layer_type = node_val
            node_properties = {}
        else:
            node_properties[node_type] = node_val.split(',')

        idx += 1
        if idx < len(raw_phenotype):
            node_type, node_val = raw_phenotype[idx].split(':')

    layers.append((layer_type, node_properties))

    return layers


def get_learning(learning):
    """
        Parses the phenotype corresponding to the learning
        Auxiliary function of the assemble_optimiser function

        Parameters
        ----------
        learning : str
            learning phenotype of the individual

        Returns
        -------
        learning_params : dict
            learning parameters
    """

    raw_learning = learning.split(' ')

    idx = 0
    learning_params = {}
    while idx < len(raw_learning):
        param_name, param_value = raw_learning[idx].split(':')
        learning_params[param_name] = param_value.split(',')
        idx += 1

    for _key_ in sorted(list(learning_params.keys())):
        if len(learning_params[_key_]) == 1:
            try:
                learning_params[_key_] = eval(learning_params[_key_][0])
            except NameError:
                learning_params[_key_] = learning_params[_key_][0]

    return learning_params
 
def assemble_network(keras_layers, input_size):
    """ 
        Maps the layers phenotype into a keras model

        Parameters
        ----------
        keras_layers : list
            output from get_layers

        input_size : tuple
            network input shape

        Returns
        -------
        model : keras.models.Model
            keras trainable model
    """

    #input layer
    inputs = tk.layers.Input(shape=input_size)

    #Create layers -- ADD NEW LAYERS HERE
    layers = []
    for layer_type, layer_params in keras_layers:
        #convolutional layer
        if layer_type == 'conv':
            conv_layer = tk.layers.Conv1D(filters=int(layer_params['num-filters'][0]),
                                                kernel_size=int(layer_params['filter-shape'][0]),
                                                strides=int(layer_params['stride'][0]),
                                                padding=layer_params['padding'][0],
                                                activation=layer_params['act'][0],
                                                use_bias=eval(layer_params['bias'][0]),
                                                kernel_initializer='he_normal',
                                                kernel_regularizer=tk.regularizers.l2(0.0005))
            layers.append(conv_layer)

        #batch-normalisation
        elif layer_type == 'batch-norm':
            #TODO - check because channels are not first
            batch_norm = tk.layers.BatchNormalization()
            layers.append(batch_norm)

        #average pooling layer
        elif layer_type == 'pool-avg':
            pool_avg = tk.layers.AveragePooling1D(pool_size=int(layer_params['kernel-size'][0]),
                                                        strides=int(layer_params['stride'][0]),
                                                        padding=layer_params['padding'][0])
            layers.append(pool_avg)

        #max pooling layer
        elif layer_type == 'pool-max':
            pool_max = tk.layers.MaxPooling1D(pool_size=int(layer_params['kernel-size'][0]),
                                                            strides=int(layer_params['stride'][0]),
                                                            padding=layer_params['padding'][0])
            layers.append(pool_max)

        #fully-connected layer
        elif layer_type == 'fc':
            fc = tk.layers.Dense(int(layer_params['num-units'][0]),
                                            activation=layer_params['act'][0],
                                            use_bias=eval(layer_params['bias'][0]),
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tk.regularizers.l2(0.0005))
            layers.append(fc)

        #dropout layer
        elif layer_type == 'dropout':
            dropout = tk.layers.Dropout(rate=min(0.5, float(layer_params['rate'][0])))
            layers.append(dropout)

        #gru layer #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
        elif layer_type == 'gru':
            gru = tk.layers.GRU(units=int(layer_params['units'][0]),
                                    activation=layer_params['act'][0],
                                    recurrent_activation=layer_params['rec_act'][0],
                                    use_bias=eval(layer_params['bias'][0]))
            layers.append(gru)

        #lstm layer #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
        elif layer_type == 'lstm':
            lstm = tk.layers.LSTM(units=int(layer_params['units'][0]),
                                        activation=layer_params['act'][0],
                                        recurrent_activation=layer_params['rec_act'][0],
                                        use_bias=eval(layer_params['bias'][0]))
            layers.append(lstm)

        #rnn #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
        elif layer_type == 'rnn':
            rnn = tk.layers.SimpleRNN(units=int(layer_params['units'][0]),
                                            activation=layer_params['act'][0],
                                            use_bias=eval(layer_params['bias'][0]))
            layers.append(rnn)

        elif layer_type == 'conv1d': #todo initializer
            conv1d = tk.layers.Conv1D(filters=int(layer_params['num-filters'][0]),
                                            kernel_size=int(layer_params['kernel-size'][0]),
                                            strides=int(layer_params['strides'][0]),
                                            padding=layer_params['padding'][0],
                                            activation=layer_params['activation'][0],
                                            use_bias=eval(layer_params['bias'][0]))
            layers.add(conv1d)


        #END ADD NEW LAYERS


    #Connection between layers
    for layer in keras_layers:
        layer[1]['input'] = list(map(int, layer[1]['input']))


    first_fc = True
    data_layers = []
    invalid_layers = []

    for layer_idx, layer in enumerate(layers):
        try:
            if len(keras_layers[layer_idx][1]['input']) == 1:
                if keras_layers[layer_idx][1]['input'][0] == -1:
                    data_layers.append(layer(inputs))
                else:
                    if keras_layers[layer_idx][0] == 'fc' and first_fc:
                        first_fc = False
                        flatten = tk.layers.Flatten()(data_layers[keras_layers[layer_idx][1]['input'][0]])
                        data_layers.append(layer(flatten))
                        continue

                    data_layers.append(layer(data_layers[keras_layers[layer_idx][1]['input'][0]]))

            else:
                #Get minimum shape: when merging layers all the signals are converted to the minimum shape
                minimum_shape = input_size[0]
                for input_idx in keras_layers[layer_idx][1]['input']:
                    if input_idx != -1 and input_idx not in invalid_layers:
                        if data_layers[input_idx].shape[-3:][0] < minimum_shape:
                            minimum_shape = int(data_layers[input_idx].shape[-3:][0])

                #Reshape signals to the same shape
                merge_signals = []
                for input_idx in keras_layers[layer_idx][1]['input']:
                    if input_idx == -1:
                        if inputs.shape[-3:][0] > minimum_shape:
                            actual_shape = int(inputs.shape[-3:][0])
                            merge_signals.append(tk.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(inputs))
                        else:
                            merge_signals.append(inputs)

                    elif input_idx not in invalid_layers:
                        if data_layers[input_idx].shape[-3:][0] > minimum_shape:
                            actual_shape = int(data_layers[input_idx].shape[-3:][0])
                            merge_signals.append(tk.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(data_layers[input_idx]))
                        else:
                            merge_signals.append(data_layers[input_idx])

                if len(merge_signals) == 1:
                    merged_signal = merge_signals[0]
                elif len(merge_signals) > 1:
                    merged_signal = tk.layers.concatenate(merge_signals)
                else:
                    merged_signal = data_layers[-1]

                data_layers.append(layer(merged_signal))
        except ValueError as e:
            data_layers.append(data_layers[-1])
            invalid_layers.append(layer_idx)
            if DEBUG:
                print(keras_layers[layer_idx][0])
                print(e)

    
    model = tk.models.Model(inputs=inputs, outputs=data_layers[-1])
    
    if DEBUG:
        model.summary()

    return model
    
    
def assemble_optimiser(learning):
        """
            Maps the learning into a keras optimiser

            Parameters
            ----------
            learning : dict
                output of get_learning

            Returns
            -------
            optimiser : keras.optimizers.Optimizer
                keras optimiser that will be later used to train the model
        """

        if learning['learning'] == 'rmsprop':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning['lr'],
                decay_steps=10000,
                decay_rate=float(learning['decay']))
            return tk.optimizers.RMSprop(learning_rate = lr_schedule,
                                            rho = float(learning['rho']))
        
        elif learning['learning'] == 'gradient-descent':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning['lr'],
                decay_steps=10000,
                decay_rate=float(learning['decay']))
            return tf.keras.optimizers.SGD(learning_rate = lr_schedule,
                                        momentum = float(learning['momentum']),
                                        nesterov = bool(learning['nesterov']))

        elif learning['learning'] == 'adam':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning['lr'],
                decay_steps=10000,
                decay_rate=float(learning['decay']))
            return tk.optimizers.Adam(learning_rate = lr_schedule,
                                         beta_1 = float(learning['beta1']),
                                         beta_2 = float(learning['beta2']))
            
def main():
    miR_label = extract_label("../dataset/tcga_mir_label.csv")
    miR_data = np.genfromtxt('../dataset/tcga_mir_rpm.csv', delimiter=',')[1:,0:-1]
    number_to_delete = abs(len(miR_label) - miR_data.shape[0])
    miR_data = miR_data[number_to_delete:,:]
    # Convert labels in number 
    num_miR_label = label_processing(miR_label)

    # Z-score 
    miR_data = normalize(miR_data)

    assert np.isnan(miR_data).sum() == 0

    #---Number of classes---#
    top_10_classes = True
    padded_data = False

    if top_10_classes:
       n_classes = 10
       miR_data, miR_label, num_miR_label = top_10_dataset(miR_data, miR_label)
    else:
        n_classes = np.unique(miR_label).size
    
    x_training, x_test, y_training, y_test = train_test_split(miR_data, num_miR_label, test_size=0.20, stratify=num_miR_label)
    x_train, x_val, y_train, y_val= train_test_split(x_training, y_training, test_size=0.10, stratify=y_training)
    
    y_train = tk.utils.to_categorical(y_train, n_classes)
    y_val = tk.utils.to_categorical(y_val, n_classes)
    y_test = tk.utils.to_categorical(y_test, n_classes)
    
    x_train = np.asarray(x_train)
    x_val = np.asarray(x_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    phenotype =  "layer:conv num-filters:145 filter-shape:49 stride:5 padding:valid act:sigmoid bias:False input:-1 layer:batch-norm input:0 layer:conv num-filters:246 filter-shape:21 stride:2 padding:valid act:relu bias:True input:1 layer:fc act:relu num-units:957 bias:False input:2 layer:dropout rate:0.5973319348751424 input:3 layer:fc act:softmax num-units:10 bias:True input:4 learning:rmsprop lr:0.0034665721628665525 rho:0.8851892507980416 decay:0.0009045137446637266 early_stop:5 batch_size:404 epochs:10000"
    model_phenotype, learning_phenotype = phenotype.split('learning:')
    learning_phenotype = 'learning:'+learning_phenotype.rstrip().lstrip()
    model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

    keras_layers = get_layers(model_phenotype)
    keras_learning = get_learning(learning_phenotype)
    batch_size = int(keras_learning['batch_size'])
    
    model = assemble_network(keras_layers, (1881, 1))
    opt = assemble_optimiser(keras_learning)
    
    model.compile(optimizer=opt,
                    loss='categorical_crossentropy',                          
                    metrics=['accuracy'])

    model.fit(x = x_train, 
                y = y_train,
                batch_size = batch_size,
                epochs = 46,
                steps_per_epoch=(x_train.shape[0]//batch_size),
                validation_data=(x_val, y_val),
                verbose = DEBUG)
    
    results = model.evaluate(x=x_test, y=y_test, batch_size = batch_size)
    print("test loss, test acc", results)
    
    # model.save('ConvNet.h5')
if __name__=="__main__":
    main()