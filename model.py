from typing import ForwardRef
from numpy.core.numeric import identity
import torch
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
    def __init__(self,epsilon,sensitivity):
        super(Model, self).__init__()

        #self.leaky_Relu = nn.LeakyReLU(negative_slope=0.3)
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.upsamplig2D = nn.UpsamplingNearest2d(scale_factor=2)
        #self.maxPool2D = nn.MaxPool2d([1,1])
        #self.sigmoid = nn.Sigmoid()
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm96 = nn.BatchNorm2d(96)
        self.batchnorm32 = nn.BatchNorm2d(32)

        self.fc1_id = nn.Linear(in_features=67, out_features=512) # for identity input
        self.fc1_ori = nn.Linear(in_features=2, out_features=512) # for orientation input
        self.fc1_em = nn.Linear(in_features=8, out_features=512) # for emotion input

        self.fc2 = nn.Linear(in_features=1536, out_features=1024) # for the concatenated 3 inputs layer
        self.fc3 = nn.Linear(in_features=1024, out_features=5*4*128) # the final linear layer before the upsampling
        
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5))
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5))
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(5,5))
        self.conv3_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3))

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5,5))
        self.conv4_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3))

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(5,5))
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3))

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(5,5))
        self.conv6_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3))

        self.conv7 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3,3))
    
    def forward(self,identity_input,orientation_input,emotion_input):
        identity_input = self.fc1_id(identity_input)
        identity_input = F.leaky_relu(identity_input, negative_slope=0.3)

        orientation_input = self.fc1_ori(orientation_input)
        orientation_input = F.leaky_relu(orientation_input, negative_slope=0.3)

        emotion_input = self.fc1_em(emotion_input)
        emotion_input = F.leaky_relu(emotion_input, negative_slope=0.3)

        
        noise = torch.from_numpy(np.random.laplace(loc=0,scale=self.sensitivity/self.epsilon, size=(1, 512)))
        noise = noise.to(identity_input.device)
        

        obfuscated_layer = identity_input + noise

        params = torch.cat((obfuscated_layer,orientation_input,emotion_input),dim=1).double()
        params = self.fc2(params)
        params = F.leaky_relu(params, negative_slope=0.3)

        x = self.fc3(params)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = x.reshape(-1,128,5,4)

        ## I is not making the upasmpling in the 2Dimensions
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv1)
        x = F.pad(x,(width,width,height,height))
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv1_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv1_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.batchnorm128(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv2_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv2_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.batchnorm128(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv3)
        x = F.pad(x,(width,width,height,height))
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv3_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv3_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.batchnorm96(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv4)
        x = F.pad(x,(width,width,height,height))
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv4_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv4_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x =self.batchnorm96(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv5)
        x = F.pad(x,(width,width,height,height))
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv5_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv5_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        x = self.batchnorm32(x)
        x = F.max_pool2d(x,kernel_size=1)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv6)
        x = F.pad(x,(width,width,height,height))
        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv6_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv6_2(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        height, width = get_paddingSizes(x, self.conv7)
        x = F.pad(x,(width,width,height,height))
        x = self.conv7(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x,0)

        return x
