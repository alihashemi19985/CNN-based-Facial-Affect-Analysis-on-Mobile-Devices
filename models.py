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
from torchsummary import summary

class Alex_Net(nn.Module):
    def __init__(self):
        super(Alex_Net,self).__init__()

        self.layer1 = nn.Sequential(
                                    nn.Conv2d(in_channels=3, out_channels=16,kernel_size=9,stride=1,padding=4),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Dropout(p =.2),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7,stride=1,padding=3),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout(p = .2),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.layer3 = nn.Sequential(

                                    nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, stride=1,padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Dropout(p = .2),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.layer4 = nn.Sequential(
                                    nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=1,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout(p = .2),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout(p = .2),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        self.flt = nn.Flatten()

        self.FC = nn.Sequential(

            
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=8),
            nn.Softmax(dim=1)
            
        )

    def forward(self,x):
         x =  self.layer1(x) 
         x = self.layer2(x)   
         x = self.layer3(x)
         x = self.layer4(x)
         x = self.layer5(x)
         x = self.flt(x)
         x = self.FC(x)
         return x 


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,strid):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin,stride=strid)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class MobileNet_Variant(nn.Module):
    def __init__(self):
        super(MobileNet_Variant,self).__init__()

        self.Standard_Conv = nn.Sequential(
                                        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()
                                )
        self.Dconv1 = nn.Sequential(
                                        depthwise_separable_conv(32,64,1),
                                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()

                                    )
        self.Dconv2 = nn.Sequential(
                                        depthwise_separable_conv(64,128,2),
                                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()

                                    )
        self.Dconv3 = nn.Sequential(
                                        depthwise_separable_conv(128,128,1),
                                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()

                                    )
        self.Dconv4 = nn.Sequential(
                                        depthwise_separable_conv(128,256,2),
                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()

                                   )
        self.Dconv5 = nn.Sequential(
                                        depthwise_separable_conv(256,256,1),
                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()

                                    )
        self.Dconv6 = nn.Sequential(
                                        depthwise_separable_conv(256,512,2),
                                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()
                                    )
        
        self.Dconv7 = self.make_Dconv7()
        self.Dconv8 = nn.Sequential(
                                        depthwise_separable_conv(512,1024,2),
                                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()
                                    )
        
        self.Dconv9 = nn.Sequential(
                                        depthwise_separable_conv(1024,1024,1),
                                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()
                                    )
        
        self.GAVP = nn.AvgPool2d(kernel_size=4)
        self.flt = nn.Flatten()
        self.FC = nn.Sequential(nn.Linear(1024,8),nn.Softmax(dim=1))
    def make_Dconv7(self):
        layers = []    
        for i in range(5):
            layers.append(nn.Sequential(depthwise_separable_conv(512,512,1),
                                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU()))
            
        return nn.Sequential(*layers)    


    def forward(self,x):
       x = self.Standard_Conv(x) 
       x = self.Dconv1(x)
       x = self.Dconv2(x)
       x = self.Dconv3(x)    
       x = self.Dconv4(x)
       x = self.Dconv5(x) 
       x = self.Dconv6(x)  
       x = self.Dconv7(x)
       x = self.Dconv8(x)
       x = self.Dconv9(x)
       x = self.GAVP(x)
       x = self.flt(x)
       x = self.FC(x) 
       return x 
