
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5) # input: (3, 32, 32) ouput: (16, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: (16, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5) # output: (32, 10, 10)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: (32, 5, 5)
        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu((self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 32*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# test
# import torch
# input = torch.rand([1, 3, 32, 32])
# model = LeNet()
# print(model)
# output = model(input)


