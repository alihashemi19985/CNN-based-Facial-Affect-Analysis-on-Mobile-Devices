
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
import torch 
import torch.nn as nn 
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader,Dataset,random_split
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from models import Alex_Net,depthwise_separable_conv,MobileNet_Variant
from torchsummary import summary 


class Checkpoint(object):
    def __init__(self):
        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)
    def save(self, acc, filename, epoch,net):
        if acc > self.best_acc:
            
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            path = os.path.join(os.path.abspath(self.folder), filename + '.pth')
            torch.save(state, path)
            self.best_acc = acc
    def load(self,clf):
        W = torch.load(r'C:\Users\Ali\Desktop\DL Papers\Face_Emotion\chekpoint\chk.pth')['net']
        clf.load_state_dict(W)
        return clf