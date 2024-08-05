# coding=utf-8
import torch.nn as nn
from torchinfo import summary

var_size = {
    'usc': {
        'in_size': 6,
        'ker_size': 6,
        'fc_size': 64*46
    }
}


class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=32, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = var_size[taskname]['fc_size']

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        return x


    def getfea(self, x):
        x = self.conv2(self.conv1(x))
        return x

class Network(nn.Module):
    def __init__(self, dataset='dsads'):
        super(Network, self).__init__()
        self.dataset = dataset
        self.var_size = {
            'dsads': {
                'in_size': 45,
                'ker_size': 9,
                'fc_size': 64*25
            },
            'pamap': {
                'in_size': 27,
                'ker_size': 9,
                'fc_size': 64*44
            },
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.var_size[self.dataset]['in_size'], out_channels=32, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = self.var_size[self.dataset]['fc_size']
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, self.var_size[self.dataset]['fc_size'])
        return x
    
    def getfea(self, x):
        x = self.conv2(self.conv1(x))
        return x
    
if __name__=='__main__':
    net = Network(dataset="dsads")
    summary(net,(1,45,1,125))
