## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # input image size is 224 * 224
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor: (32,220,220) -  (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # 32 inputs, 64 output channels, 5x5 square convolution kernel
        # reduced input after maxpooling: (32,110,110)
        self.conv2 = nn.Conv2d(32, 64 ,5) # output tensor: (64,106,106) -  (W-F)/S + 1 = (110-5)/1 + 1 = 106
        
        # 64 inputs, 128 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (64,53,53)
        self.conv3 = nn.Conv2d(64, 128 ,3) # output tensor: (128,51,51) -  (W-F)/S + 1 = (53-3)/1 + 1 = 51
        
        # 128 inputs, 256 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (128,25,25)
        self.conv4 = nn.Conv2d(128, 256 ,3) # output tensor: (256,23,23) -  (W-F)/S + 1 = (25-3)/1 + 1 = 23
        
        # 256 inputs, 512 output channels, 1x1 square convolution kernel
        # reduced input after maxpooling: (256,12,12)
        self.conv5 = nn.Conv2d(256, 512 ,1) # output tensor: (512,12,12) -  (W-F)/S + 1 = (12-1)/1 + 1 = 12
        
        # maxpool layer
        # pool with kerne_size=2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout layer
        # dropout probability p= 0.2
        self.drop = nn.Dropout(p=0.2)
        
        # fully-connected layers
        self.fc1 =  nn.Linear(12800, 1024)
        self.fc2 =  nn.Linear(1024, 512)
        self.fc3 =  nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        #print ('Conv1: ', x.size())
        x = self.pool(x)
        #print ('Pool1: ', x.size())
        #x = self.drop(x)

        x = F.relu(self.conv2(x))
        #print ('Conv2: ', x.size())
        x = self.pool(x)
        #print ('Pool2: ', x.size())
        #x= self.drop(x)
        
        x = F.relu(self.conv3(x))
        #print ('Conv3: ', x.size())
        x = self.pool(x)
        #print ('Pool3:', x.size())
        #x = self.drop(x)

        x = F.relu(self.conv4(x))
        #print ('Conv4: ', x.size())
        x = self.pool(x)
        #print ('Pool4:', x.size())
        #x = self.drop(x)

        x = F.relu(self.conv5(x))
        #print ('Conv4: ', x.size())
        x = self.pool(x)
        #print ('Pool5:', x.size())
        #x = self.drop(x)
        
        # flatten the layer to generate input for linear layers
        x = x.view(x.size(0), -1)
        #print ('Flattend: ', x.size())
 
        x = F.relu(self.fc1(x))
        #x = self.drop(x)
 
        x = F.relu(self.fc2(x))
        #x = self.drop(x)        
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 224 by 224 pixels

        # input image size is 224 * 224
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor: (32,220,220) -  (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # 32 inputs, 64 output channels, 5x5 square convolution kernel
        # reduced input after maxpooling: (32,110,110)
        self.conv2 = nn.Conv2d(32, 64 ,5) # output tensor: (64,106,106) -  (W-F)/S + 1 = (110-5)/1 + 1 = 106
        
        # 64 inputs, 128 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (64,53,53)
        self.conv3 = nn.Conv2d(64, 128 ,3) # output tensor: (128,51,51) -  (W-F)/S + 1 = (53-3)/1 + 1 = 51
        
        # 128 inputs, 256 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (128,25,25)
        self.conv4 = nn.Conv2d(128, 256 ,3) # output tensor: (256,23,23) -  (W-F)/S + 1 = (25-3)/1 + 1 = 23
        
        # 256 inputs, 512 output channels, 1x1 square convolution kernel
        # reduced input after maxpooling: (256,12,12)
        self.conv5 = nn.Conv2d(256, 512 ,1) # output tensor: (512,12,12) -  (W-F)/S + 1 = (12-1)/1 + 1 = 12
        
        # maxpool layer
        # pool with kerne_size=2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout layer
        # dropout probability p= 0.2
        self.drop = nn.Dropout(p=0.2)
        
        # fully-connected layers
        self.fc1 =  nn.Linear(12800, 1024)
        self.fc2 =  nn.Linear(1024, 512)
        self.fc3 =  nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
        
        
class Net3(nn.Module):

    def __init__(self):

        super(Net3, self).__init__()

               # input image size is 224 * 224
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor: (32,220,220) -  (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # 32 inputs, 64 output channels, 5x5 square convolution kernel
        # reduced input after maxpooling: (32,110,110)
        self.conv2 = nn.Conv2d(32, 64 ,5) # output tensor: (64,106,106) -  (W-F)/S + 1 = (110-5)/1 + 1 = 106
        
        # 64 inputs, 128 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (64,53,53)
        self.conv3 = nn.Conv2d(64, 128 ,3) # output tensor: (128,51,51) -  (W-F)/S + 1 = (53-3)/1 + 1 = 51
        
        # 128 inputs, 256 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (128,25,25)
        self.conv4 = nn.Conv2d(128, 256 ,3) # output tensor: (256,23,23) -  (W-F)/S + 1 = (25-3)/1 + 1 = 23
        
        # 256 inputs, 512 output channels, 1x1 square convolution kernel
        # reduced input after maxpooling: (256,12,12)
        self.conv5 = nn.Conv2d(256, 512 ,1) # output tensor: (512,12,12) -  (W-F)/S + 1 = (12-1)/1 + 1 = 12
        
        # maxpool layer
        # pool with kerne_size=2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout layer
        # dropout probability p= 0.2
        self.drop = nn.Dropout(p=0.2)
        
        # fully-connected layers
        self.fc1 =  nn.Linear(12800, 1024)
        self.fc2 =  nn.Linear(1024, 512)
        self.fc3 =  nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
class Net4(nn.Module):

    def __init__(self):
        super(Net4, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # input image size is 224 * 224
        
       # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor: (32,220,220) -  (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # 32 inputs, 64 output channels, 5x5 square convolution kernel
        # reduced input after maxpooling: (32,110,110)
        self.conv2 = nn.Conv2d(32, 64 ,5) # output tensor: (64,106,106) -  (W-F)/S + 1 = (110-5)/1 + 1 = 106
        
        # 64 inputs, 128 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (64,53,53)
        self.conv3 = nn.Conv2d(64, 128 ,3) # output tensor: (128,51,51) -  (W-F)/S + 1 = (53-3)/1 + 1 = 51
        
        # 128 inputs, 256 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (128,25,25)
        self.conv4 = nn.Conv2d(128, 256 ,3) # output tensor: (256,23,23) -  (W-F)/S + 1 = (25-3)/1 + 1 = 23
        
        # 256 inputs, 512 output channels, 1x1 square convolution kernel
        # reduced input after maxpooling: (256,12,12)
        self.conv5 = nn.Conv2d(256, 512 ,1) # output tensor: (512,12,12) -  (W-F)/S + 1 = (12-1)/1 + 1 = 12
        
        # maxpool layer
        # pool with kerne_size=2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout layer
        # dropout probability p= 0.2
        self.drop = nn.Dropout(p=0.2)
        
        # fully-connected layers
        self.fc1 =  nn.Linear(80000, 1024)
        self.fc2 =  nn.Linear(1024, 512)
        self.fc3 =  nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        #print ('Conv1: ', x.size())
        x = self.pool(x)
        #print ('Pool1: ', x.size())
        #x = self.drop(x)

        x = F.relu(self.conv2(x))
        #print ('Conv2: ', x.size())
        x = self.pool(x)
        #print ('Pool2: ', x.size())
        #x= self.drop(x)
        
        x = F.relu(self.conv3(x))
        #print ('Conv3: ', x.size())
        x = self.pool(x)
        #print ('Pool3:', x.size())
        #x = self.drop(x)
        
        # flatten the layer to generate input for linear layers
        x = x.view(x.size(0), -1)
        #print ('Flattend: ', x.size())
 
        x = F.relu(self.fc1(x))
        #x = self.drop(x)
 
        x = F.relu(self.fc2(x))
        #x = self.drop(x)        
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Net5(nn.Module):

    def __init__(self):
        super(Net5, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # input image size is 224 * 224
        
       # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor: (32,220,220) -  (W-F)/S + 1 = (224-5)/1 + 1 = 220
        
        # 32 inputs, 64 output channels, 5x5 square convolution kernel
        # reduced input after maxpooling: (32,110,110)
        self.conv2 = nn.Conv2d(32, 64 ,5) # output tensor: (64,106,106) -  (W-F)/S + 1 = (110-5)/1 + 1 = 106
        
        # 64 inputs, 128 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (64,53,53)
        self.conv3 = nn.Conv2d(64, 128 ,3) # output tensor: (128,51,51) -  (W-F)/S + 1 = (53-3)/1 + 1 = 51
        
        # 128 inputs, 256 output channels, 3x3 square convolution kernel
        # reduced input after maxpooling: (128,25,25)
        self.conv4 = nn.Conv2d(128, 256 ,3) # output tensor: (256,23,23) -  (W-F)/S + 1 = (25-3)/1 + 1 = 23
        
        # 256 inputs, 512 output channels, 1x1 square convolution kernel
        # reduced input after maxpooling: (256,12,12)
        self.conv5 = nn.Conv2d(256, 512 ,1) # output tensor: (512,12,12) -  (W-F)/S + 1 = (12-1)/1 + 1 = 12
        
        
        # maxpool layer
        # pool with kerne_size=2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout layer
        # dropout probability p= 0.2
        self.drop = nn.Dropout(p=0.2)
        
        # fully-connected layers
        self.fc1 =  nn.Linear(80000, 1024)
        self.fc2 =  nn.Linear(1024, 512)
        self.fc3 =  nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        #print ('Conv1: ', x.size())
        x = self.pool(x)
        #print ('Pool1: ', x.size())
        #x = self.drop(x)

        x = F.relu(self.conv2(x))
        #print ('Conv2: ', x.size())
        x = self.pool(x)
        #print ('Pool2: ', x.size())
        #x= self.drop(x)
        
        x = F.relu(self.conv3(x))
        #print ('Conv3: ', x.size())
        x = self.pool(x)
        #print ('Pool3:', x.size())
        #x = self.drop(x)

        
        # flatten the layer to generate input for linear layers
        x = x.view(x.size(0), -1)
        #print ('Flattend: ', x.size())
 
        x = F.relu(self.fc1(x))
        x = self.drop(x)
 
        x = F.relu(self.fc2(x))
        #x = self.drop(x)        
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

