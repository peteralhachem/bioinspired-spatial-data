import torch
import csv
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import TensorDataset, DataLoader
import pprint
import tensorflow.keras as tk
import tensorflow as tf

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

def build_dataloader(miR_data, num_miR_label, padded_data, batch_size=404):

    if padded_data:
        miR_data = add_pad_data(miR_data)

    train_data, val_data, train_label, val_label = train_test_split(miR_data, num_miR_label, test_size=0.20, random_state=42)
    #train_label = tk.utils.to_categorical(train_label, num_classes=10, dtype="float32")

    miR_train = torch.Tensor(train_data)
    miR_train_label = torch.Tensor(train_label)
    miR_dataset_train = TensorDataset(miR_train, miR_train_label)

    miR_val = torch.Tensor(val_data)
    miR_val_label = torch.Tensor(val_label)
    miR_dataset_val = TensorDataset(miR_val, miR_val_label)

    train_loader = DataLoader(miR_dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(miR_dataset_val, batch_size=batch_size)

    if padded_data:
        num_inputs = train_data.shape[2] ** 2
    else:
        num_inputs = train_data.shape[1]

    return num_inputs, train_loader, test_loader

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

def getData():

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    miR_label = extract_label("../../dataset/tcga_mir_label.csv")
    miR_data = np.genfromtxt('../../dataset/tcga_mir_rpm.csv', delimiter=',')[1:,0:-1]
    number_to_delete = abs(len(miR_label) - miR_data.shape[0])
    miR_data = miR_data[number_to_delete:,:]
    # Convert labels in number 
    num_miR_label = label_processing(miR_label)

    # Z-score 
    miR_data = normalize(miR_data)

    assert np.isnan(miR_data).sum() == 0

    #---Number of classes---#
    top_10_classes = True

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

    return x_train, y_train, x_val, y_val, x_test, y_test