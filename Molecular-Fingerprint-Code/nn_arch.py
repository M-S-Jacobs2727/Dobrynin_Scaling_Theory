<<<<<<< HEAD
#import numpy as np
from torch import flatten
#from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, LogSoftmax, BatchNorm2d, Dropout, Flatten
#from torch.optim import Adam, SGD
#from torch.nn import transforms
import json

class Net(Module):
    #def __init__(self, numChannels, classes):
    def __init__(self):
        '''
        Initializes a neural network architecture        
        '''
        super(Net, self).__init__()

        #with open("channel_layers.json") as f:
        #    num_layers = json.load(f)

        #self.conv1 = Conv2d(in_channels=numChannels, out_channels=num_layers['cnn_channels_1'], kernel_size=(5,5), padding=1)
        self.conv1 = Conv2d(in_channels=1, out_channels=6,kernel_size=3)
        # output of first convolutional layer is 5x28x28 activation map
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2)
        # output is 6x1414 map

        #self.conv2 = Conv2d(in_channels=num_layers['cnn_channels_1'], out_channels=num_layers['cnn_channels_2'], kernel_size=(5, 5), padding=0)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # output of second convolutional layer is 16x12x12 activation map
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2)
        # output is 16x6x6 map

        self.fc1 = Linear(in_features=16*6*6, out_features=200)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=200, out_features=3)
        self.LogSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.LogSoftmax(x)

        return output
=======
#import numpy as np
from torch import flatten
#from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, LogSoftmax, BatchNorm2d, Dropout, Flatten
#from torch.optim import Adam, SGD
#from torch.nn import transforms
import json

class Net(Module):
    #def __init__(self, numChannels, classes):
    def __init__(self):
        '''
        Initializes a neural network architecture        
        '''
        super(Net, self).__init__()

        #with open("channel_layers.json") as f:
        #    num_layers = json.load(f)

        #self.conv1 = Conv2d(in_channels=numChannels, out_channels=num_layers['cnn_channels_1'], kernel_size=(5,5), padding=1)
        self.conv1 = Conv2d(in_channels=1, out_channels=6,kernel_size=3)
        # output of first convolutional layer is 5x28x28 activation map
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2)
        # output is 6x1414 map

        #self.conv2 = Conv2d(in_channels=num_layers['cnn_channels_1'], out_channels=num_layers['cnn_channels_2'], kernel_size=(5, 5), padding=0)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # output of second convolutional layer is 16x12x12 activation map
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2)
        # output is 16x6x6 map

        self.fc1 = Linear(in_features=16*6*6, out_features=200)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=200, out_features=3)
        self.LogSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.LogSoftmax(x)

        return output
>>>>>>> e55fca987c7909fbfce2b0223005c5fd029fdc5b
