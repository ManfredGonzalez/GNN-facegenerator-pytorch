import argparse
import yaml
from typing import ForwardRef
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import RaFDDataset
import numpy as np
from MSLELoss import MSLELoss
from RMSLELoss import RMSLELoss
from norm_distance_metric import Distance_metric
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime

from model import Model
from utils import CustomDataParallel,boolean_string
import traceback
import os
import cv2

reverse_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            np.array,
        ])
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

def train(opt):
    

    '''project = 'fourth_NO_lr_scheduler'
    log_path = 'E:/datosmanfred/Slovennian_research/GNN-weights'
    saved_path = 'E:/datosmanfred/Slovennian_research/GNN-weights'''
     
    
    project = opt.project
    log_path = opt.log_path
    saved_path = opt.saved_path
    num_epochs = opt.num_epochs
    learningRate = opt.lr
    use_cuda = opt.use_cuda
    save_interval = opt.save_interval
    es_patience = opt.es_patience
    es_min_delta = opt.es_min_delta

    #raFDDataset = RaFDDataset(opt.data_path,(1024,1280))
    raFDDataset = RaFDDataset(opt.data_path,(512,640))
    #raFDDataset = RaFDDataset('E:/datosmanfred/Slovennian_research/RafD_frontal_536',(256,320))

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(raFDDataset,
                                            batch_size=opt.batch_size,
                                            shuffle=opt.shuffle_ds,
                                            num_workers=opt.num_workers)

    best_loss = 1000.0
    best_epoch = 0


    #log_path = f'E:/datosmanfred/Slovennian_research/GNN-weights/{project}/tensorboard/'
    #saved_path = f'E:/datosmanfred/Slovennian_research/GNN-weights/{project}'
    log_path = f'{log_path}{project}/tensorboard/'
    saved_path = f'{saved_path}{project}'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)
    model = Model(opt.epsilon, opt.sensitivity, loc_laplace=opt.loc_laplace).double()

    if opt.load_weights is not None:
        weights_path = opt.load_weights

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), learningRate)

    last_step = 0
    step = max(0, last_step)
    num_iter_per_epoch = len(data_loader)
    # if something wrong happens, catch and save the last weights
    writer = SummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
    if opt.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True, factor=0.5, min_lr=1e-6)
    
    imax=1
    imin=0
    preds = []
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
                    n = torch.flatten(images).size(dim=0)
                    
                    if opt.loss_function == 'msle':
                        lossFunction = MSLELoss()
                    elif opt.loss_function == 'rmsle':
                        lossFunction = RMSLELoss()
                    elif opt.loss_function == 'mse':
                        lossFunction = nn.MSELoss()
                    else:
                        lossFunction = MSLELoss()
                    
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
                cv2.imwrite(os.path.join(saved_path,'best.jpg'),reverse_preprocess(preds[0].cpu().detach()))
                with open(os.path.join(saved_path, f"best_weights.txt"), "a") as my_file: 
                    my_file.write(f"Epoch:{epoch} / Step: {step} / Loss: {best_loss}\n") 
            cv2.imwrite(os.path.join(saved_path,'last.jpg'),reverse_preprocess(preds[0].cpu().detach()))

            if epoch - best_epoch > es_patience > 0:
                print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                break
            if opt.lr_scheduler:
                scheduler.step(np.mean(epoch_loss))
    except KeyboardInterrupt:
            save_checkpoint(model, f'GNN_trained_weights_last.pth')
            writer.close()
    writer.close()
#               Section for handling parameters from user
#--------------------------------------------------------------------------------------------------------------------
class Params:
    """Read file with parameters"""
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('GNN Pytorch')
    parser.add_argument('-p', '--project', type=str, default='GNN-pytorch')
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=100) # Number of steps between saving
    parser.add_argument('--es_min_delta', type=float, default=0.0) # Early stopping's parameter: minimum change loss to qualify as an improvement
    parser.add_argument('--es_patience', type=int, default=0) # Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.
    parser.add_argument('--data_path', type=str, default='datasets/') # the root folder of dataset
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None) # whether to load weights from a checkpoint, set None to initialize, set last to load last checkpoint
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--shuffle_ds', type=boolean_string, default=True)
    parser.add_argument('--use_cuda', type=boolean_string, default=True) 
    parser.add_argument('--epsilon', type=int, default=50)
    parser.add_argument('--sensitivity', type=int, default=1) 
    parser.add_argument('--lr_scheduler', type=boolean_string, default=True)
    parser.add_argument('--loss_function', type=str, default='msle')
    parser.add_argument('--loc_laplace', type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39]) 
    opt = get_args()
    train(opt)