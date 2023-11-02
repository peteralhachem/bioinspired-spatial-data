from cnn_model import ConvNet, generate_dependent_parameters
import torch, torch.nn as nn
from utils import extract_label, label_processing, top_10_dataset, build_dataloader, normalize, set_seed
import numpy as np
import nni
import random
import torch.optim as optim
import os

def trainCNN(model, device, train_dataloader, test_dataloader, num_epochs = 173):

    model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()  
        
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            optimizer.zero_grad()  
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {accuracy:.2f}%")
        nni.report_intermediate_result(accuracy)
            
    nni.report_final_result(accuracy)
    
def main():
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device ", device)
    current_working_directory = os.getcwd()

    # print output to the console
    print(current_working_directory)
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

    params = {
        'w1': random.randint(2, 256),
        'w2': random.randint(2, 256),
        'w3': random.randint(2, 256),
        'wd1': random.randint(4, 64),
        'h1': random.randint(4, 64),
    }
    
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    
    h2, wd2, h3, wd3 = generate_dependent_parameters(num_inputs, params['wd1'], params['h1'])
    
    print(f"h2: {h2}, wd2: {wd2}, h3: {h3}, wd3: {wd3}")

    conv_net = ConvNet(num_inputs, w1=params['w1'], wd1=params['wd1'], h1=params['h1'], w2=params['w2'], 
                       wd2=wd2, h2=h2, w3=params['w3'], wd3=wd3, h3=h3, w4=n_classes)    
    
    trainCNN(conv_net, device=device, train_dataloader=train_loader, test_dataloader=test_loader, num_epochs=50)

if __name__ == '__main__':
    main()
    
