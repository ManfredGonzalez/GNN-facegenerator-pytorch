from typing import ForwardRef
from numpy.core.numeric import identity
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import RaFDDataset

def get_paddingSizes(x, layer):
    #supposing that the feature map has the shape (batch_size,channels,height,width)
    width = int((((x.shape[3]-1)-(x.shape[3]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    height = int((((x.shape[2]-1)-(x.shape[2]-layer.kernel_size[0]))*(1 / layer.stride[0]))/2)
    return height, width


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.leaky_Relu = nn.LeakyReLU(negative_slope=0.3)
        self.upsamplig2D = nn.UpsamplingNearest2d(scale_factor=2)
        self.maxPool2D = nn.MaxPool2d([1,1])
        self.sigmoid = nn.Sigmoid()

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
        identity_input = self.leaky_Relu(identity_input)

        orientation_input = self.fc1_ori(orientation_input)
        orientation_input = self.leaky_Relu(orientation_input)

        emotion_input = self.fc1_em(emotion_input)
        emotion_input = self.leaky_Relu(emotion_input)

        params = torch.cat((identity_input,orientation_input,emotion_input)).double()
        params = self.fc2(params)
        params = self.leaky_Relu(params)

        x = self.fc3(params)
        x = self.leaky_Relu(x)
        x = x.reshape(128,5,4)

        x = torch.unsqueeze(x,0)
        ## I is not making the upasmpling in the 2Dimensions
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv1)
        x = F.pad(x,(width,width,height,height))
        x = self.conv1(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv1_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv1_2(x)
        x = self.leaky_Relu(x)

        x = nn.BatchNorm2d(128).double()(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv2(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv2_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv2_2(x)
        x = self.leaky_Relu(x)

        x = nn.BatchNorm2d(128).double()(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv3)
        x = F.pad(x,(width,width,height,height))
        x = self.conv3(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv3_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv3_2(x)
        x = self.leaky_Relu(x)

        x = nn.BatchNorm2d(96).double()(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv4)
        x = F.pad(x,(width,width,height,height))
        x = self.conv4(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv4_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv4_2(x)
        x = self.leaky_Relu(x)

        x = nn.BatchNorm2d(96).double()(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv5)
        x = F.pad(x,(width,width,height,height))
        x = self.conv5(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv5_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv5_2(x)
        x = self.leaky_Relu(x)

        x = nn.BatchNorm2d(32).double()(x)
        x = self.maxPool2D(x)
        x = self.upsamplig2D(x)

        height, width = get_paddingSizes(x, self.conv6)
        x = F.pad(x,(width,width,height,height))
        x = self.conv6(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv6_2)
        x = F.pad(x,(width,width,height,height))
        x = self.conv6_2(x)
        x = self.leaky_Relu(x)

        height, width = get_paddingSizes(x, self.conv7)
        x = F.pad(x,(width,width,height,height))
        x = self.conv7(x)
        x = self.sigmoid(x)

        x = torch.squeeze(x,0)

        return x
