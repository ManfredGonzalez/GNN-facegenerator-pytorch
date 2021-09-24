from numpy.core.numeric import identity
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from instance_pytorch import RaFDDataset
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        num_kernels = [128, 128, 96, 96, 32, 32, 16]
        initial_shape=(5,4)
        height, width = initial_shape
        #indentity shape = torch.Size([67])
        #emotion shape = torch.Size([8])
        #orientation shape = torch.Size([2])
        self.fc1_id = nn.Linear(in_features=67, out_features=512) # for identity input
        self.fc1_ori = nn.Linear(in_features=2, out_features=512) # for orientation input
        self.fc1_em = nn.Linear(in_features=8, out_features=512) # for emotion input

        self.fc2 = nn.Linear(in_features=1536, out_features=1024) # for the concatenated 3 inputs layer
        self.fc3 = nn.Linear(in_features=1024, out_features=5*4*128) # the final linear layer before the upsampling
        
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), padding='same')
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same')

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), padding='same')
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same')

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(5,5), padding='same')
        self.conv3_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding='same')

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5,5), padding='same')
        self.conv4_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding='same')

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(5,5), padding='same')
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
raFDDataset = RaFDDataset('datasets/RafD_frontal_536',(128,128))

# own DataLoader
data_loader = torch.utils.data.DataLoader(raFDDataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=0)

batch = next(iter(data_loader))
orientations, identities, emotions, images = batch


model = Model()
identity_input = identities[0].float()
orientation_input = orientations[0].float()
emotion_input = emotions[0].float()

identity_input = model.fc1_id(identity_input)
identity_input = F.leaky_relu(identity_input)

orientation_input = model.fc1_ori(orientation_input)
orientation_input = F.leaky_relu(orientation_input)

emotion_input = model.fc1_em(emotion_input)
emotion_input = F.leaky_relu(emotion_input)

params = torch.cat((identity_input,orientation_input,emotion_input))
params = model.fc2(params)
params = F.leaky_relu(params)

x = model.fc3(params)
x = F.leaky_relu(x)
x = x.reshape(128,5,4)

## I is not making the upasmpling in the 2Dimensions
x = F.upsample(input=x,scale_factor=[2,2,2])
x = model.conv1(x)
x = F.leaky_relu(x)
x = model.conv1_2(x)
x = F.leaky_relu(x)
print(x)