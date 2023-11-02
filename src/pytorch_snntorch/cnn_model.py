import torch
import torch.nn as nn
import random
import math

class ConvNet(nn.Module):
    def __init__(self, input_size, w1, wd1, h1, w2, wd2, h2, w3, wd3, h3, w4):
        super().__init__()
        conv1_out = ((input_size - 1 * (wd1 - 1) -1) + 1)
        conv1_out = int(conv1_out)
        
        s1 = (((conv1_out - 1 * (h1 - 1) -1)/h1) + 1)
        s1 = int(s1)
        
        conv2_out = ((s1 - 1 * (wd2 - 1)-1) + 1)
        conv2_out = int(conv2_out)
        
        s2 = (((conv2_out - 1 * (h2 -1 ) -1) / h2 ) + 1)
        s2 = int(s2)
        
        conv3_out = ((s2 - 1 * (wd3 - 1)-1) + 1)
        conv3_out = int(conv3_out)
        
        s3_dec = (((conv3_out - 1 * (h3 - 1 ) -1) / h3) + 1)
        if s3_dec < 1:
            s3 = 1
        else:
            s3 = math.floor(s3_dec)

        if s3 == 0:
            s3 = 1

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=w1, kernel_size=wd1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=h1)
        
        self.conv2 = nn.Conv1d(in_channels=w1, out_channels=w2, kernel_size=wd2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=h2)
        
        self.conv3 = nn.Conv1d(in_channels=w2, out_channels=w3, kernel_size=wd3)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=h3)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(s3*w3, w4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# w1, w2, w3, w4 independent
# h1, wd1 independent

def generate_dependent_parameters(input_size, wd1, h1):
    conv1_out = int((input_size - 1 * (wd1 - 1)-1) + 1)
    s1 = int(((conv1_out - 1 * (h1 -1 ) -1) / h1 ) + 1)
    
    h2 = random.randint(2, s1)
    wd2 = random.randint(2, s1)
    
    conv2_out = int((s1 - 1 * (wd2 - 1)-1) + 1)
    s2 = int(((conv2_out - 1 * (h2 - 1) -1) / h2 ) + 1)
    while s2 < 2:
        h2 = random.randint(2, s1) 
        wd2 = random.randint(2, s1) 
        conv2_out = int((s1 - 1 * (wd2 - 1)-1) + 1)
        s2 = int(((conv2_out - 1 * (h2 - 1) -1) / h2 ) + 1)
    if s2 == 2:
        h3 = 1
        wd3 = 2
    else:
        wd3 = random.randint(2, s2)
        h3 = random.randint(2, wd3)
        s3 = math.floor(((s2 - wd3 + 1)/h3)+1)
        if s2 == wd3 or (s2 - wd3 < h3): 
            h3 = 1
        
        """print("start loop")
        while h3 >= s3:
            h3 = random.randint(2, s2) 
            wd3 = random.randint(2, s2)
            s3 = math.floor(((s2 - wd3 + 1)/h3)+1)
        print("end loop")"""
    return h2, wd2, h3, wd3
    
def main():
    w1 = random.randint(2, 256)
    w2 = random.randint(2, 256)
    w3  = random.randint(2, 256)
    w4 = random.randint(2, 256)
    wd1 = random.randint(4, 64)
    h1 = random.randint(4, 64)
    
    h2, wd2, h3, wd3 = generate_dependent_parameters(1881, wd1, h1)
    model = ConvNet(input_size=1881, w1=w1, wd1=wd1, h1=h1, w2=w2, wd2=wd2, h2=h2, w3=w3, wd3=wd3, h3=h3, w4=10)
    
    # net with the best hyperparams found by paper EA.
    # model = ConvNet(input_size=1046, w1=36, wd1=206, h1=175, w2=251, wd2=41, h2=4, w3=2, wd3=133, h3=7, w4=2)
    x = torch.rand((1,1,1881))
    out = model(x)
    
if __name__=="__main__":
    main()