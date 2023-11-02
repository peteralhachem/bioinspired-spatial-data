# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from fast_denser.utilities.datasets.svhn import load_svhn
from fast_denser.utilities.datasets.cifar import load_cifar
from fast_denser.utilities.datasets.tiny_imagenet import load_tiny_imagenet
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import pprint
import math
import csv 
import scipy
from multiprocessing import Pool
import tensorflow as tf
import contextlib
import sys

def extract_label(file_name, verbose=False):
    data = {}
    label = []
    with open(file_name, "r") as fin:
        reader = csv.reader(fin, delimiter=',')
        first = True
        for row in reader:
            lbl = row[2]
            if first or "TARGET" in lbl:
                first = False
                continue
            lbl = lbl.replace("TCGA-","")

            label.append(lbl)
            if lbl in data.keys():
                data[lbl] += 1 
            else:
                data[lbl] = 1
    if verbose:
        print(f"Number of classes in the dataset = {len(data)}")
        pprint.pprint(data, indent=4)

    return label

def create_dictionary(labels):
    dictionary = {}
    class_names = np.unique(labels)
    for i, name in enumerate(class_names):
        dictionary[name] = i
    return dictionary

def label_processing(labels):
    new_miRna_label = []
    dictionary = create_dictionary(labels)
    for i in labels:
        new_miRna_label.append(dictionary[i])
    return new_miRna_label

def add_pad_data(data):
  miR_data = data
  c_int = math.ceil(np.sqrt(len(miR_data[0])))
  pad = c_int ** 2 - len(miR_data[0])
  pad_width = (0, pad)

  padded_miR_data = np.zeros((miR_data.shape[0], miR_data.shape[1] + pad_width[1]))

  for i in range(len(miR_data)):
    padded_miR_data[i] = np.pad(miR_data[i], pad_width, mode='constant')

  # reshape shape[1] into (c_int, c_int)

  dim = int(np.sqrt(len(padded_miR_data[0])))
  padded_miR_data = padded_miR_data.reshape((padded_miR_data.shape[0],1, dim, dim))

  return padded_miR_data

def top_10_dataset(miR_data, miR_label):
  occ = dict({k: 0 for k in set(miR_label)})

  for i in range(len(miR_label)):
    occ[miR_label[i]] += 1

  top_10_class = sorted(occ, key=occ.get,reverse=True)[:10]

  list_top_10_train = []
  list_top_10_labels = []

  for i in range(len(miR_label)):
    if miR_label[i] in top_10_class:
      list_top_10_labels.append(miR_label[i])

  for i in range(miR_data.shape[0]):
    if miR_label[i] in top_10_class:
      list_top_10_train.append(miR_data[i])

  miR_data_reduced = np.stack(list_top_10_train, axis=0)
  miR_label_reduced = list_top_10_labels

  num_miR_label_reduced = label_processing(miR_label_reduced)

  return miR_data_reduced, miR_label_reduced, num_miR_label_reduced

def normalize(data, method='zscore'):
    if method == "zscore":
        return scipy.stats.zscore(data, axis=1)
   
    # log2 normalization
    elif method=="log2":
        data = data + abs(np.min(data)) + 0.001
        return np.log2(data)
    
    # normalization between [0, 255]
    else:
       return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

#dataset paths - change if the path is different
SVHN = 'fast_denser/utilities/datasets/data/svhn'
TINY_IMAGENET = 'fast_denser/utilities/datasets/data/tiny-imagenet-200'

def prepare_data(x_train, y_train, x_test, y_test, n_classes=10):
    """
        Split the data into independent sets

        Parameters
        ----------
        x_train : np.array
            training instances
        y_train : np.array
            training labels 
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels


        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of the model
                                          on unseen data
    """

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train.reshape((-1, 32, 32, 3))
    x_test = x_test.reshape((-1, 32, 32, 3))

    evo_x_train, x_val, evo_y_train, y_val = train_test_split(x_train, y_train,
                                                              test_size = 7000,
                                                              stratify = y_train)

    evo_x_val, evo_x_test, evo_y_val, evo_y_test = train_test_split(x_val, y_val,
                                                                    test_size = 3500,
                                                                    stratify = y_val)


    evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
    evo_y_val = keras.utils.to_categorical(evo_y_val, n_classes)

    dataset = {
        'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
        'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
        'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
        'x_test': x_test, 'y_test': y_test
    }

    return dataset



def resize_data(args):
    """
        Resize the dataset 28 x 28 datasets to 32x32

        Parameters
        ----------
        args : tuple(np.array, (int, int))
            instances, and shape of the reshaped signal

        Returns
        -------
        content : np.array
            reshaped instances
    """

    import tensorflow as tf

    content, shape = args
    content = content.reshape(-1, 28, 28, 1)

    if shape != (28, 28):
        content = tf.image.resize(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    content = tf.image.grayscale_to_rgb(tf.constant(content))
    
    return content.numpy()



def load_dataset(dataset, shape=(32,32)):
    """
        Load a specific dataset

        Parameters
        ----------
        dataset : str
            dataset to load

        shape : tuple(int, int)
            shape of the instances

        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of the model
                                          on unseen data
    """


    if dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        n_classes = 10

        x_train = 255-x_train
        x_test = 255-x_test

        num_pool_workers=1 
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(resize_data, [(x_train, shape)])
            x_train = pool_results.get()[0]

        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(resize_data, [(x_test, shape)])
            x_test = pool_results.get()[0]

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        n_classes = 10

        num_pool_workers=1 
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(resize_data, [(x_train, shape)])
            x_train = pool_results.get()[0]

        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(resize_data, [(x_test, shape)])
            x_test = pool_results.get()[0]
        
    #255, unbalanced
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn(SVHN)
        n_classes = 10

    #255, 50000, 10000
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar(10)
        n_classes = 10

    #255, 50000, 10000
    elif dataset == 'cifar100-fine':
        x_train, y_train, x_test, y_test = load_cifar(100, 'fine')
        n_classes = 100

    elif dataset == 'cifar100-coarse':
        x_train, y_train, x_test, y_test = load_cifar(100, 'coarse')
        n_classes = 20

    elif dataset == 'tiny-imagenet':
        x_train, y_train, x_test, y_test = load_tiny_imagenet(TINY_IMAGENET, shape)
        n_classes = 200

    elif dataset == 'biomarkers':
        miR_label = extract_label("fast_denser/utilities/datasets/data/tcga_mir_label.csv")
        miR_data = np.genfromtxt('fast_denser/utilities/datasets/data/tcga_mir_rpm.csv', delimiter=',')[1:,0:-1]
        number_to_delete = abs(len(miR_label) - miR_data.shape[0])
        miR_data = miR_data[number_to_delete:,:]
        
        miR_data = normalize(miR_data)
        miR_data, miR_label, num_miR_label = top_10_dataset(miR_data, miR_label)
        # Convert labels in number 
        num_miR_label = label_processing(miR_label)
        x_train, x_test, y_train, y_test = train_test_split(miR_data, num_miR_label, test_size=0.20)
        n_classes = 10
    else:
        print('Error: the dataset is not valid')
        sys.exit(-1)
    
    #dataset = prepare_data(x_train, y_train, x_test, y_test, n_classes)
    evo_x_train, x_val, evo_y_train, y_val = train_test_split(x_train, y_train,
                                                              stratify = y_train)

    evo_x_val, evo_x_test, evo_y_val, evo_y_test = train_test_split(x_val, y_val,
                                                                    stratify = y_val)
    evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
    evo_y_val = keras.utils.to_categorical(evo_y_val, n_classes)
    dataset = {
        'evo_x_train': np.asarray(evo_x_train), 'evo_y_train': np.asarray(evo_y_train),
        'evo_x_val': np.asarray(evo_x_val), 'evo_y_val': np.asarray(evo_y_val),
        'evo_x_test': np.asarray(evo_x_test), 'evo_y_test': np.asarray(evo_y_test),
        'x_test': np.asarray(x_test), 'y_test': np.asarray(y_test)
    }
    
    return dataset
