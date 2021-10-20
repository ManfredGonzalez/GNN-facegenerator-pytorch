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
from tensorboardX import SummaryWriter
import datetime

from model import Model
from utils import CustomDataParallel
import traceback
import os

def save_checkpoint(model, name,saved_path):
    '''
    Save current weights of the model.
    params
    :model (torch.model) -> model to save.
    :name (string) -> filename to save this model.
    '''
    if isinstance(model, CustomDataParallel):
        torch.save(model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(saved_path, name))


raFDDataset = RaFDDataset('E:/datosmanfred/Slovennian_research/RafD_frontal_536',(256,320))

# own DataLoader
data_loader = torch.utils.data.DataLoader(raFDDataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)

num_epochs = 500
learningRate = 1e-3
use_cuda = True
best_loss = 1000.0
best_epoch = 0
es_patience = 0
es_min_delta = 0.0
save_interval = 100
project = 'fourth_NO_lr_scheduler'
log_path = 'E:/datosmanfred/Slovennian_research/GNN-weights'
saved_path = 'E:/datosmanfred/Slovennian_research/GNN-weights'


log_path = f'E:/datosmanfred/Slovennian_research/GNN-weights/{project}/tensorboard/'
saved_path = f'E:/datosmanfred/Slovennian_research/GNN-weights/{project}'
os.makedirs(log_path, exist_ok=True)
os.makedirs(saved_path, exist_ok=True)
model = Model().double()

if use_cuda:
    model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), learningRate)

last_step = 0
step = max(0, last_step)
num_iter_per_epoch = len(data_loader)
# if something wrong happens, catch and save the last weights
writer = SummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5, min_lr=1e-6)
try:
    for epoch in range(num_epochs):
        total_loss = 0
        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue
        epoch_loss = []
        progress_bar = tqdm(data_loader)
        for iter, batch in enumerate(progress_bar):
            if iter < step - last_epoch * num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                input_data, images = batch
                preds = []
                lossFunction = RMSLELoss()
                #lossFunction = nn.MSELoss()
                if use_cuda:
                    identities = input_data[1].cuda()
                    orientations = input_data[0].cuda()
                    emotions = input_data[2].cuda()
                    images = images.cuda()
                    lossFunction = lossFunction.cuda()
                else:
                    identities = input_data[1]
                    orientations = input_data[0]
                    emotions = input_data[2]
                

                    

                preds = model(identities,orientations,emotions)

                loss = lossFunction(preds,images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #store the loss for later use in the scheduler
                epoch_loss.append(float(loss))
                # record in the logs
                #----------------------------
                # record in the logs
                #----------------------------
                progress_bar.set_description(
                    'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Total loss: {:.5f}'.format(
                        step, epoch, num_epochs, iter + 1, num_iter_per_epoch, loss.item()))
                writer.add_scalars('Loss', {'train': loss}, step)

                # log learning_rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('learning_rate', current_lr, step)

                step += 1
                # save the model
                
                # save the model
                if step % save_interval == 0 and step > 0:
                    save_checkpoint(model, f'last_weights.pth',saved_path)
                    #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_trained_weights.pth')
                    print('checkpoint...')
                
            
            except Exception as e:
                        print('[Error]', traceback.format_exc())
                        print(e)
                        continue
        # Early stopping
        if loss + es_min_delta < best_loss:
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(model, f'best_weights.pth',saved_path)
            with open(os.path.join(saved_path, f"best_weights.txt"), "a") as my_file: 
                my_file.write(f"Epoch:{epoch} / Step: {step} / Loss: {best_loss}\n") 

        if epoch - best_epoch > es_patience > 0:
            print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
            break
        
        scheduler.step(np.mean(epoch_loss))
except KeyboardInterrupt:
        save_checkpoint(model, f'GNN_trained_weights_last.pth')
        writer.close()
writer.close()
