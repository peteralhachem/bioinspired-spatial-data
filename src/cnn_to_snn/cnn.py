import torch, torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=145, kernel_size=49, stride=5, padding='valid', bias=False)
            self.act1 = nn.Sigmoid()
            
            self.bn1 = nn.BatchNorm1d(num_features=145)
            
            self.conv2 = nn.Conv1d(in_channels=145, out_channels=246, kernel_size=21, stride=2, padding='valid', bias=True)
            self.act2 = nn.ReLU()
            
            self.flatten = nn.Flatten()
            
            self.fc1 = nn.Linear(in_features=42804, out_features=957, bias=False)
            self.act3 = nn.ReLU()
            
            self.drop = nn.Dropout(p=0.5973319348751424)
            self.fc2 = nn.Linear(in_features=957, out_features=10, bias=True)
            self.act3 = nn.Softmax(dim=1)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act3(x)
        return x