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
import math

def get_paddingSizes(x, layer):
    #supposing that the feature map has the shape (batch_size,channels,height,width)
    width = int((((x.shape[3]-1)-(x.shape[3]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    height = int((((x.shape[2]-1)-(x.shape[2]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    return height, width

class Model(nn.Module):
    def __init__(self,epsilon=0,sensitivity=1,loc_laplace=0):
        super(Model, self).__init__()

        #self.leaky_Relu = nn.LeakyReLU(negative_slope=0.3)
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.loc_laplace = loc_laplace
        self.upsamplig2D = nn.UpsamplingNearest2d(scale_factor=2)

        self.fc1_id = nn.Linear(in_features=67, out_features=512) # for identity input
        self.fc1_ori = nn.Linear(in_features=2, out_features=512) # for orientation input
        self.fc1_em = nn.Linear(in_features=8, out_features=512) # for emotion input

        self.fc1_id_2 = nn.Linear(in_features=512, out_features=512) # for identity input

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
        ##New layers
        '''self.upconv6 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32)
        )'''
        '''self.upconv7 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5,5), padding='same'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(16)
        )'''

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
            identity_input = self.fc1_id_2(identity_input)
            params = torch.cat((identity_input,orientation_input,emotion_input),dim=1).double()
        else:
            n = identity_input.size(dim=1)
            ## Way 1
            #imax = torch.max(identity_input[identity_input < 1]).item()
            #imin = torch.min(identity_input[identity_input > 0]).item()
            ## Way 2
            imax = torch.max(identity_input).item()
            imin = torch.min(identity_input).item()

            

            scale_parameter = (((n*(imax-imin))/self.epsilon))
            #sensitivity = np.exp(self.epsilon)
            sensitivity = 1
            value = sensitivity/scale_parameter
            #noise = torch.from_numpy(np.random.laplace(loc=0,scale=value, size=(1, n)))
            noise = torch.from_numpy(np.random.laplace(loc=0,scale=scale_parameter, size=(1, n)))
            noise = noise.to(identity_input.device)
            #snap any out-of-bounds noisy value back to the nearest valid value
            valid_noise = noise[noise < imax]
            valid_noise = valid_noise[valid_noise > imin]
            ## Way 2
            if valid_noise.size(0) != 0:
                noise_max = torch.max(valid_noise).item()
                noise[noise > noise_max] = noise_max
                noise_min = torch.min(valid_noise).item()
                noise[noise < noise_min] = noise_min

            obfuscated_input = identity_input + noise ##Adding the noise values to ther features

            identity_input = self.fc1_id_2(obfuscated_input)

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
        #x = self.upconv6(x)
        #x = self.upconv7(x)

        
        x = F.max_pool2d(x,kernel_size=1)
        x = self.upsamplig2D(x)

        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.conv6_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        
        x = self.conv7(x)
        x = torch.sigmoid(x)

        return x
