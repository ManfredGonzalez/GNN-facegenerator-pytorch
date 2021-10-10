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
from MSLELoss import RMSLELoss
from tqdm import tqdm

from model import Model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = Model().double()



raFDDataset = RaFDDataset('/home/manfred/RafD_frontal_536',(256,320))

# own DataLoader
data_loader = torch.utils.data.DataLoader(raFDDataset,
                                          batch_size=3,
                                          shuffle=True,
                                          num_workers=0)

num_epochs = 100
learningRate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), learningRate)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(data_loader,"Batch Processing"):
        
        orientations, identities, emotions, images = batch
        preds = []
        lossFunction = RMSLELoss()
        for i in range(images.shape[0]):
            identity_input = identities[i].double()
            orientation_input = orientations[i].double()
            emotion_input = emotions[i].double()

            preds.append(model(identity_input,orientation_input,emotion_input))
        preds = torch.stack(preds)

        loss = lossFunction(preds,images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
print(total_loss)