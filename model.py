from typing import ForwardRef
from numpy.core.numeric import identity
import torch
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import RaFDDataset
import numpy as np

def get_paddingSizes(x, layer):
    #supposing that the feature map has the shape (batch_size,channels,height,width)
    width = int((((x.shape[3]-1)-(x.shape[3]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    height = int((((x.shape[2]-1)-(x.shape[2]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    return height, width


class Model(nn.Module):
    def __init__(self,epsilon=0,sensitivity=1):
        super(Model, self).__init__()

        #self.leaky_Relu = nn.LeakyReLU(negative_slope=0.3)
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.upsamplig2D = nn.UpsamplingNearest2d(scale_factor=2)

        self.fc1_id = nn.Linear(in_features=67, out_features=512) # for identity input
        self.fc1_ori = nn.Linear(in_features=2, out_features=512) # for orientation input
        self.fc1_em = nn.Linear(in_features=8, out_features=512) # for emotion input

        self.fc2 = nn.Linear(in_features=1536, out_features=1024) # for the concatenated 3 inputs layer
        self.fc3 = nn.Linear(in_features=1024, out_features=5*4*128) # the final linear layer before the upsampling
        
        self.upconv1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(128)
        )
        self.upconv2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(128)
        )
        self.upconv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(96)
        )
        self.upconv4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(96)
        )
        self.upconv5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32)
        )

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(5,5), padding='same')
        self.conv6_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), padding='same')

        self.conv7 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3,3), padding='same')
    
    def forward(self,identity_input,orientation_input,emotion_input):
        identity_input = self.fc1_id(identity_input)
        identity_input = F.leaky_relu(identity_input, negative_slope=0.3)

        orientation_input = self.fc1_ori(orientation_input)
        orientation_input = F.leaky_relu(orientation_input, negative_slope=0.3)

        emotion_input = self.fc1_em(emotion_input)
        emotion_input = F.leaky_relu(emotion_input, negative_slope=0.3)

        if self.epsilon == 0:
            noise = torch.from_numpy(np.random.laplace(loc=0,scale=self.sensitivity/self.epsilon, size=(1, 512)))
            noise = noise.to(identity_input.device)
            obfuscated_layer = identity_input + noise
            params = torch.cat((obfuscated_layer,orientation_input,emotion_input),dim=1).double()
        else:
            params = torch.cat((identity_input,orientation_input,emotion_input),dim=1).double()
        
        params = self.fc2(params)
        params = F.leaky_relu(params, negative_slope=0.3)

        x = self.fc3(params)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = x.reshape(-1,128,5,4)

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)

        
        x = F.max_pool2d(x,kernel_size=1)
        x = self.upsamplig2D(x)

        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.conv6_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        
        x = self.conv7(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x,0)

        return x
