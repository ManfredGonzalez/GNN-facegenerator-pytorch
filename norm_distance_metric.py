import torch
import torch.nn as nn
import cv2
import numpy as np

class Distance_metric(nn.Module):
    def __init__(self):
        super(Distance_metric,self).__init__()
    def forward(self,im1,im2):
        im1 = torch.flatten(im1)
        im2 = torch.flatten(im2)

        imax = torch.max(torch.Tensor((torch.max(im1),torch.max(im2))))## get the maximum value from both images
        imin = torch.min(torch.Tensor((torch.min(im1),torch.min(im2))))## get the minimum value from both images

        distance = torch.sum(torch.abs(im1-im2)/(imax-imin)) # normalize the distance between elements
        
        n = im1.size(dim=0)
        result = distance/n #normalization to account for models
        return result,imax,imin
