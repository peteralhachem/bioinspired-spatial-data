import torch 
import torch.nn as nn
import torch.optim as optim
from snntorch import utils, surrogate
import numpy as np
import snntorch.functional as SF
from cnn_model import ConvNet
from nni_snn_search import get_cnn_dimension, build_layer
from utils import build_dataloader, normalize, extract_label, label_processing, top_10_dataset, set_seed
import nni
import os
import argparse
import json

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lightweight', action=argparse.BooleanOptionalAction)
    args = parser.parse_args(args=args) 
    return args

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

def trainCNN(model, device, train_dataloader, test_dataloader, num_epochs = 50, lightweight=False):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = -np.inf
    for epoch in range(num_epochs):
        model.train()  
        
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.float)
            optimizer.zero_grad()  
            
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_dataloader)}")

        model.eval()  
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                #inputs = inputs.unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {accuracy:.2f}%")
        if accuracy >= best_acc:
            best_acc = accuracy
            print(f"New best accuracy found! Value: {best_acc:.2f}. Saving model..")
            if not lightweight:
                torch.save(model.state_dict(), 'cnn_best_acc_trained_model.pt')
            else:
                torch.save(model.state_dict(), 'cnn_light_trained_model.pt')

def test_accuracy(train_loader, net, num_steps, device):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    train_loader = iter(train_loader)
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)

        
        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

  return acc/total

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def main():
    args = parse_args()
    
    set_seed(42)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
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

    num_inputs, train_loader, test_loader = build_dataloader(miR_data, num_miR_label, padded_data, batch_size=128)

    if not args.lightweight:
        with open("../best_hyperparams/cnn_params_best.json", "r") as f:
            params_cnn = json.load(f)
    else:
        with open("../best_hyperparams/cnn_params_light.json", "r") as f:
            params_cnn = json.load(f)
        
    s1, s2, s3 = get_cnn_dimension(num_inputs, params_cnn)
    
    net = ConvNet(num_inputs, params_cnn['w1'], params_cnn['wd1'],params_cnn['h1'], 
                        params_cnn['w2'], params_cnn['wd2'], params_cnn['h2'],
                        params_cnn['w3'], params_cnn['wd3'], params_cnn['h3'], n_classes)
    print("CNN number of parameters: ", count_parameters(net))
    
    if not args.lightweight:
        if not os.path.isfile("../trained_models/cnn_best_acc_trained_model.pt"):
            trainCNN(net, device, train_loader, test_loader, num_epochs=200, lightweight=args.lightweight)
            
        state_dict = torch.load("../trained_models/cnn_best_acc_trained_model.pt")
    else:
        if not os.path.isfile("../trained_models/cnn_light_trained_model.pt"):
            trainCNN(net, device, train_loader, test_loader, num_epochs=200, lightweight=args.lightweight)
            
        state_dict = torch.load("../trained_models/cnn_light_trained_model.pt")
    
    params_snn = {
        'beta': 0.7,
        'num_steps': 50,
        'learning_rate': 0.0015,
        'neuron_type': 'lapicque'
    }
    
    optimized_params = nni.get_next_parameter()
    params_snn.update(optimized_params)
    
    grad = surrogate.fast_sigmoid()
    
    snn_model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=params_cnn['w1'], kernel_size=params_cnn['wd1']),
        nn.MaxPool1d(kernel_size=params_cnn['h1']),
        build_layer(params_snn['neuron_type'], params_snn['beta'], grad, s1),
        nn.Conv1d(in_channels=params_cnn['w1'], out_channels=params_cnn['w2'], kernel_size=params_cnn['wd2']),
        nn.MaxPool1d(kernel_size=params_cnn['h2']),
        build_layer(params_snn['neuron_type'], params_snn['beta'], grad, s2),
        nn.Conv1d(in_channels=params_cnn['w2'], out_channels=params_cnn['w3'], kernel_size=params_cnn['wd3']),
        nn.MaxPool1d(kernel_size=params_cnn['h3']),
        build_layer(params_snn['neuron_type'], params_snn['beta'], grad, s3),
        nn.Flatten(),
        nn.Linear(s3*params_cnn['w3'], n_classes),
        build_layer(params_snn['neuron_type'], params_snn['beta'], grad, n_classes, output=True)
    )
    
    with torch.no_grad():
        snn_model[0].weight = torch.nn.Parameter(state_dict['conv1.weight'])
        snn_model[0].bias = torch.nn.Parameter(state_dict['conv1.bias'])
        snn_model[3].weight = torch.nn.Parameter(state_dict['conv2.weight'])
        snn_model[3].bias = torch.nn.Parameter(state_dict['conv2.bias'])
        snn_model[6].weight = torch.nn.Parameter(state_dict['conv3.weight'])
        snn_model[6].bias = torch.nn.Parameter(state_dict['conv3.bias'])
        snn_model[10].weight = torch.nn.Parameter(state_dict['fc1.weight'])
        snn_model[10].bias = torch.nn.Parameter(state_dict['fc1.bias'])
    
    snn_model.to(device)
    print(f"SNN number of parameters: {count_parameters(snn_model)}")
          
    optimizer = torch.optim.Adam(snn_model.parameters(), lr=params_snn['learning_rate'], betas=(0.9, 0.999))
    
    loss_fn = SF.ce_count_loss()
        
    num_epochs = 30
    curr_acc = -np.inf
    
    # Outer training loop
    for epoch in range(num_epochs):
        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            snn_model.train()
            spk_rec, _ = forward_pass(snn_model, params_snn['num_steps'], data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        epoch_acc = test_accuracy(test_loader, snn_model, params_snn['num_steps'], device)
        nni.report_intermediate_result(epoch_acc)
        if epoch_acc >= curr_acc:
            curr_acc = epoch_acc
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {epoch_acc*100:.2f}%")
    nni.report_final_result(curr_acc)
    
if __name__=="__main__":
    main()
    