import math
import torch 
import torch.nn as nn
import snntorch as snn
from snntorch import utils, surrogate
import numpy as np
import snntorch.functional as SF
from utils import build_dataloader, normalize, extract_label, label_processing, top_10_dataset, set_seed
import nni
import argparse
import json

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lightweight', action=argparse.BooleanOptionalAction)
    args = parser.parse_args(args=args) 
    return args

# consider turning into a class s.t. dictionary params can be parsed at
# __init__
# and never touched again
def TBPTT(
    net,
    dataloader,
    optimizer,
    criterion,
    num_steps=False,  # only specified if time-static
    time_var=True,  # specifies if data is time_varying
    time_first=True,
    regularization=False,
    device="cpu",
    K=1,
):
    """Truncated backpropagation through time. LIF layers require parameter
    ``init_hidden = True``.
    Weight updates are performed every ``K`` time steps.

    Example::

        import snntorch as snn
        import snntorch.functional as SF
        from snntorch import utils
        from snntorch import backprop
        import torch
        import torch.nn as nn

        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

        net = nn.Sequential(nn.Flatten(),
                            nn.Linear(784,500),
                            lif1,
                            nn.Linear(500, 10),
                            lif2).to(device)

        device = torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
        num_steps = 100

        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4,
        betas=(0.9, 0.999))
        loss_fn = SF.mse_count_loss()
        reg_fn = SF.l1_rate_sparsity()

        # train_loader is of type torch.utils.data.DataLoader
        # if input data is time-static, set time_var=False, and specify
        # num_steps.
        # if input data is time-varying, set time_var=True and do not
        # specify num_steps.
        # backprop is automatically applied every K=40 time steps

        for epoch in range(5):
            loss = backprop.RTRL(net, train_loader, optimizer=optimizer,
            criterion=loss_fn, num_steps=num_steps, time_var=False,
            regularization=reg_fn, device=device, K=40)


    :param net: Network model (either wrapped in Sequential container or as a
        class)
    :type net: torch.nn.modules.container.Sequential

    :param dataloader: DataLoader containing data and targets
    :type dataloader: torch.utils.data.DataLoader

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion from snntorch.functional, e.g.,
        snn.functional.mse_count_loss()
    :type criterion: snn.functional.LossFunctions

    :param num_steps: Number of time steps. Does not need to be
        specified if ``time_var=True``.
    :type num_steps: int, optional

    :param time_var: Set to ``True`` if input data is time-varying
        [T x B x dims]. Otherwise, set to false if input data is time-static
        [B x dims], defaults to ``True``
    :type time_var: Bool, optional

    :param time_first: Set to ``False`` if first dimension of data is not
        time [B x T x dims] AND must also be permuted to [T x B x dims],
        defaults to ``True``
    :type time_first: Bool, optional

    :param regularization: Option to add a regularization term to the loss
        function
    :type regularization: snn.functional regularization function, optional

    :param device: Specify either "cuda" or "cpu", defaults to "cpu"
    :type device: string, optional

    :param K: Number of time steps to process per weight update, defaults
        to ``1``
    :type K: int, optional

    :return: return average loss for one epoch
    :rtype: torch.Tensor

    """

    if num_steps and time_var:
        raise ValueError(
            "``num_steps`` should not be specified if time_var is ``True``. "
            "When using time-varying input data, the size of the time-first "
            "dimension of each batch is automatically used as ``num_steps``."
        )

    if num_steps is False and time_var is False:
        raise ValueError(
            "``num_steps`` must be specified if ``time_var`` is ``False``. "
            "When using time-static input data, ``num_steps`` must be "
            "passed in."
        )

    if num_steps and K > num_steps:
        raise ValueError("``K`` must be less than or equal to ``num_steps``.")

    if time_var is False and time_first is False:
        raise ValueError(
            "``time_first`` should not be specified if data is not "
            "time-varying, i.e., ``time_var`` is ``False``."
        )

    # triggers global variables is_lapicque etc for neurons_dict
    # redo reset in training loop
    utils.reset(net=net)

    neurons_dict = {
        utils.is_lapicque: snn.Lapicque,
        utils.is_leaky: snn.Leaky,
        utils.is_synaptic: snn.Synaptic,
        utils.is_alpha: snn.Alpha,
        utils.is_rleaky: snn.RLeaky,
        utils.is_rsynaptic: snn.RSynaptic,
        utils.is_sconv2dlstm: snn.SConv2dLSTM,
        utils.is_slstm: snn.SLSTM,
    }

    # element 1: if true: spk, if false, mem
    # element 2: if true: time_varying_targets
    criterion_dict = {
        "mse_membrane_loss": [
            False,
            True,
        ],  # if time_var_target is true, need a flag to let mse_mem_loss
        # know when to re-start iterating targets from
        "ce_max_membrane_loss": [False, False],
        "ce_rate_loss": [True, False],
        "ce_count_loss": [True, False],
        "mse_count_loss": [True, False],
        "ce_latency_loss": [True, False],
        "mse_temporal_loss": [True, False],
        "ce_temporal_loss": [True, False],
    }  # note: when using mse_count_loss, the target spike-count should be
    # for a truncated time, not for the full time

    reg_dict = {"l1_rate_sparsity": True}

    # acc_dict = {
    #     SF.accuracy_rate : [False, False, False, True]
    # }

    time_var_targets = False
    counter = len(criterion_dict)
    for criterion_key in criterion_dict:
        if criterion_key == criterion.__name__:
            loss_spk, time_var_targets = criterion_dict[
                criterion_key
            ]  # m: mem, s: spk // s: every step, e: end
            if time_var_targets:
                time_var_targets = criterion.time_var_targets  # check this
        counter -= 1
    if counter:  # fix the print statement
        raise TypeError(
            "``criterion`` must be one of the loss functions in "
            "``snntorch.functional``: e.g., 'mse_membrane_loss', "
            "'ce_max_membrane_loss', 'ce_rate_loss' etc."
        )

    if regularization:
        for reg_item in reg_dict:
            if reg_item == regularization.__name__:
                # m: mem, s: spk // s: every step, e: end
                reg_spk = reg_dict[reg_item]

    num_return = utils._final_layer_check(net)  # number of outputs

    step_trunc = 0  # ranges from 0 to K, resetting every K time steps
    K_count = 0
    loss_trunc = 0  # reset every K time steps
    loss_avg = 0
    iter_count = 0

    mem_rec_trunc = []
    spk_rec_trunc = []

    net = net.to(device)

    data_iterator = iter(dataloader)
    for data, targets in data_iterator:
        iter_count += 1
        net.train()
        data = data.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        if time_var:
            if time_first:
                num_steps = data.size(0)
            else:
                num_steps = data.size(1)

            if K is False:
                K_flag = K
            if K_flag is False:
                K = num_steps

        utils.reset(net)

        for step in range(num_steps):
            if num_return == 2:
                if time_var:
                    if time_first:
                        spk, mem = net(data[step])
                    else:
                        spk, mem = net(data.transpose(1, 0)[step])
                else:
                    spk, mem = net(data)

            elif num_return == 3:
                if time_var:
                    if time_first:
                        spk, _, mem = net(data[step])
                    else:
                        spk, _, mem = net(data.transpose(1, 0)[step])
                else:
                    spk, _, mem = net(data)

            elif num_return == 4:
                if time_var:
                    if time_first:
                        spk, _, _, mem = net(data[step])
                    else:
                        spk, _, _, mem = net(data.transpose(1, 0)[step])
                else:
                    spk, _, _, mem = net(data)

            # else:  # assume not an snn.Layer returning 1 val
            #     if time_var:
            #         spk = net(data[step])
            #     else:
            #         spk = net(data)
            #     spk_rec.append(spk)

            spk_rec_trunc.append(spk)
            mem_rec_trunc.append(mem)

            step_trunc += 1
            if step_trunc == K:
                # spk_rec += spk_rec_trunc # test
                # mem_rec += mem_rec_trunc # test

                spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
                mem_rec_trunc = torch.stack(mem_rec_trunc, dim=0)

                # loss_spk is True if input to criterion is spk;
                # reg_spk is True if input to reg is spk

                # catch case for time_varying_targets?
                if time_var_targets:
                    if loss_spk:
                        loss = criterion(
                            spk_rec_trunc,
                            targets[int(K_count * K) : int((K_count + 1) * K)],
                        )
                    else:
                        loss = criterion(
                            mem_rec_trunc,
                            targets[int(K_count * K) : int((K_count + 1) * K)],
                        )
                else:
                    if loss_spk:
                        loss = criterion(spk_rec_trunc, targets.long())
                    else:
                        loss = criterion(mem_rec_trunc, targets.long())

                if regularization:
                    if reg_spk:
                        loss += regularization(spk_rec_trunc)
                    else:
                        loss += regularization(mem_rec_trunc)

                loss_trunc += loss
                loss_avg += loss / (num_steps / K)

                optimizer.zero_grad()
                loss_trunc.backward()
                optimizer.step()

                for neuron in neurons_dict:
                    if neuron:
                        neurons_dict[neuron].detach_hidden()
                        # detach_hidden --> _reset_hidden

                K_count += 1
                step_trunc = 0
                loss_trunc = 0
                spk_rec_trunc = []
                mem_rec_trunc = []

        if (step == num_steps - 1) and (num_steps % K):
            spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
            mem_rec_trunc = torch.stack(mem_rec_trunc, dim=0)

            if time_var_targets:
                idx1 = K_count * K
                idx2 = K_count * K + num_steps % K
                if loss_spk:
                    loss = criterion(
                        spk_rec_trunc,
                        targets[int(idx1) : int(idx2)],
                    )
                else:
                    loss = criterion(
                        mem_rec_trunc,
                        targets[int(idx1) : int(idx2)],
                    )
            else:
                if loss_spk:
                    loss = criterion(spk_rec_trunc, targets)
                else:
                    loss = criterion(mem_rec_trunc, targets)

            if regularization:
                if reg_spk:
                    loss += regularization(spk_rec_trunc)
                else:
                    loss += regularization(mem_rec_trunc)

            loss_trunc += loss
            loss_avg += loss / int(num_steps % K)

            optimizer.zero_grad()
            loss_trunc.backward()
            optimizer.step()

            K_count = 0
            step_trunc = 0
            loss_trunc = 0
            spk_rec_trunc = []
            mem_rec_trunc = []

            for neuron in neurons_dict:
                if neuron:
                    neurons_dict[neuron].detach_hidden()

    return loss_avg / iter_count  # , spk_rec, mem_rec

def BPTT(
    net,
    dataloader,
    optimizer,
    criterion,
    num_steps=False,
    time_var=True,
    time_first=True,
    regularization=False,
    device="cpu",
):
    """Backpropagation through time. LIF layers require parameter
    ``init_hidden = True``.
    A forward pass is applied for each time step while the loss accumulates.
    The backward pass and parameter update is only applied at the end of
    each time step sequence.
    BPTT is equivalent to TBPTT for the case where ``num_steps = K``.

    Example::

        import snntorch as snn
        import snntorch.functional as SF
        from snntorch import utils
        from snntorch import backprop
        import torch
        import torch.nn as nn

        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

        net = nn.Sequential(nn.Flatten(),
                            nn.Linear(784,500),
                            lif1,
                            nn.Linear(500, 10),
                            lif2).to(device)

        device = torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
        num_steps = 100

        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4,
        betas=(0.9, 0.999))
        loss_fn = SF.mse_count_loss()
        reg_fn = SF.l1_rate_sparsity()


        # train_loader is of type torch.utils.data.DataLoader
        # if input data is time-static, set time_var=False, and specify
        # num_steps.
        # if input data is time-varying, set time_var=True and do not
        # specify num_steps.

        for epoch in range(5):
            loss = backprop.RTRL(net, train_loader, optimizer=optimizer,
            criterion=loss_fn, num_steps=num_steps, time_var=False,
            regularization=reg_fn, device=device)


    :param net: Network model (either wrapped in Sequential container or as
        a class)
    :type net: torch.nn.modules.container.Sequential

    :param dataloader: DataLoader containing data and targets
    :type dataloader: torch.utils.data.DataLoader

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion from snntorch.functional, e.g.,
        snn.functional.mse_count_loss()
    :type criterion: snn.functional.LossFunctions

    :param num_steps: Number of time steps. Does not need to be specified if
        ``time_var=True``.
    :type num_steps: int, optional

    :param time_var: Set to ``True`` if input data is time-varying
        [T x B x dims]. Otherwise, set to false if input data is time-static
        [B x dims], defaults to ``True``
    :type time_var: Bool, optional

    :param time_first: Set to ``False`` if first dimension of data is not
        time [B x T x dims] AND must also be permuted to [T x B x dims],
        defaults to ``True``
    :type time_first: Bool, optional

    :param regularization: Option to add a regularization term to the loss
        function
    :type regularization: snn.functional regularization function, optional

    :param device: Specify either "cuda" or "cpu", defaults to "cpu"
    :type device: string, optional

    :return: return average loss for one epoch
    :rtype: torch.Tensor
    """

    #  Net requires hidden instance variables rather than global instance
    #  variables for TBPTT
    return TBPTT(
        net,
        dataloader,
        optimizer,
        criterion,
        num_steps,
        time_var,
        time_first,
        regularization,
        device,
        K=num_steps,
    )

def test_accuracy(data_loader, net, device, population_code=False, num_classes=False):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      utils.reset(net)
      spk_rec, _ = net(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=num_classes) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
        
      total += spk_rec.size(1)

  return acc/total

def trainSNN(net, device, learning_rate, num_epochs, num_steps, population_coding, n_classes, train_loader, test_loader):
    curr_acc = -np.inf
    net = net.to(device)
    print(f"Net number of parameters: {count_parameters(net)}")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=population_coding, num_classes=n_classes)
    
    for epoch in range(num_epochs):
        net.train()
        avg_loss = BPTT(net, train_loader, num_steps=num_steps,
                    optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        
        epoch_acc = test_accuracy(test_loader, net, device, population_code=population_coding, num_classes=n_classes if population_coding else False)
        nni.report_intermediate_result(epoch_acc)
        if epoch_acc >= curr_acc:
            curr_acc = epoch_acc
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {epoch_acc*100:.2f}%")
    
    nni.report_final_result(curr_acc)
    return curr_acc

def build_layer(layer_type, beta, grad, num_outputs, output=False):
    if layer_type=="leaky":
        if output==True:
            return snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=output, threshold=0.4) 
        else:
            return snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, threshold=0.4)
        
    if layer_type=="lapicque":
        if output==True:
            return snn.Lapicque(beta=beta, spike_grad=grad, init_hidden=True, output=output, threshold=0.4) 
        else:
            return snn.Lapicque(beta=beta, spike_grad=grad, init_hidden=True, threshold=0.4) 
    
    if layer_type=="rleaky":
        if output==True:
            return snn.RLeaky(beta=beta, spike_grad=grad, init_hidden=True, output=output, linear_features=num_outputs, threshold=0.4)
        else:
            return snn.RLeaky(beta=beta, spike_grad=grad, init_hidden=True, linear_features=num_outputs, threshold=0.4) 
        
def get_cnn_dimension(input_size, params_cnn):
    conv1_out = ((input_size - 1 * (params_cnn['wd1'] - 1) -1) + 1)
    conv1_out = int(conv1_out)
    
    s1 = (((conv1_out - 1 * (params_cnn['h1'] - 1) -1)/params_cnn['h1']) + 1)
    s1 = int(s1)
    
    conv2_out = ((s1 - 1 * (params_cnn['wd2'] - 1)-1) + 1)
    conv2_out = int(conv2_out)
    
    s2 = (((conv2_out - 1 * (params_cnn['h2'] -1 ) -1) / params_cnn['h2'] ) + 1)
    s2 = int(s2)
    
    conv3_out = ((s2 - 1 * (params_cnn['wd3'] - 1)-1) + 1)
    conv3_out = int(conv3_out)
    
    s3_dec = (((conv3_out - 1 * (params_cnn['h3'] - 1 ) -1) / params_cnn['h3']) + 1)
    if s3_dec < 1:
        s3 = 1
    else:
        s3 = math.floor(s3_dec)

    if s3 == 0:
        s3 = 1
    return s1, s2, s3

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
    
    params_snn = {
        'beta': 0.7,
        'population_coding': True,
        'num_steps': 50,
        'neurons_per_classes':75,
        'learning_rate': 1e-3,
        'neuron_type': 'leaky'
    }
    
    optimized_params = nni.get_next_parameter()
    params_snn.update(optimized_params)
    
    grad = surrogate.fast_sigmoid()
    
    num_outputs = n_classes if params_snn['population_coding']==False else params_snn['neurons_per_classes']*n_classes

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
        nn.Linear(s3*params_cnn['w3'], num_outputs),
        build_layer(params_snn['neuron_type'], params_snn['beta'], grad, num_outputs, output=True)
    )
    
    trainSNN(snn_model, device , params_snn['learning_rate'], 30, params_snn['num_steps'], params_snn['population_coding'], n_classes, train_loader, test_loader)
    
if __name__=="__main__":
    main()