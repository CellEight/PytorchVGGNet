import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGE(nn.Module):
    """ The VGG-D CNN architecture as described in the 2014 paper 
        "Very Deep Convolutional Networks for Large-Scale Image Recognition"
        by Karen Simonyan, Andrew Zisserman. See Table 1 of the paper.""" 
    def __init__(self, n_class):
        super().__init__() 
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv2_2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_1 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_4 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_1 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_3 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_4 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_1 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_3 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_4 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.fc1   = nn.Linear(25088,4096)
        self.fc2   = nn.Linear(4096,4096)
        self.fc3   = nn.Linear(4096,n_class)
        self.pool  = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.relu(self.conv4_4(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.relu(self.conv5_4(x))
        x = self.pool(x)
        x = x.view(-1,25088)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
