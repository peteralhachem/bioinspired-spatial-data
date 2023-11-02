import torch, torch.nn as nn
import snntorch as snn


def build_layer(layer_type, grad, beta, num_outputs, last_layer=False):
    
    if layer_type == "leaky": 
        if last_layer:
            return snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True) 
        else: 
            return snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True) 
        
    elif layer_type == "lapicque": 
        if last_layer:
            return snn.Lapicque(beta=beta, spike_grad=grad, init_hidden=True, output=True, threshold=0.4) 
        else:
            return snn.Lapicque(beta=beta, spike_grad=grad, init_hidden=True, threshold=0.4) 
    
    else: #rleaky type 
        if last_layer:
            return snn.RLeaky(beta=beta, spike_grad=grad, init_hidden=True, output=True, linear_features=num_outputs, threshold=0.4)
        else:
            return snn.RLeaky(beta=beta, spike_grad=grad, init_hidden=True, linear_features=num_outputs, threshold=0.4) 

def build_net(layer_types, dict_params, grad, num_inputs, num_outputs):
    layers = [nn.Flatten(), nn.Linear(num_inputs, dict_params['num_hidden'])]
    for i, layer in enumerate(layer_types):
        if i != 0:
            if i < len(layer_types) - 1:
                layers.append(nn.Linear(dict_params['num_hidden'], dict_params['num_hidden']))
            else:
                layers.append(nn.Linear(dict_params['num_hidden'], num_outputs))
            
        if i < len(layer_types)-1:
            layers.append(build_layer(layer, grad, dict_params['beta'], dict_params['num_hidden']))
        else:
            layers.append(build_layer(layer, grad, dict_params['beta'], num_outputs, last_layer=True))
        
    return nn.Sequential(*layers)


