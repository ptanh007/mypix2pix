import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LeakyReLU
from torch.nn import LogSoftmax
from torch import flatten

class Conv(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)        
        return x

class TransConv(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(TransConv, self).__init__()
        
        self.transconv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_planes)        
        
    def forward(self, x):
        x = self.transconv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
               
        return x

class Encoder(Module):
    def __init__(self):       
        super(Encoder, self).__init__()
        
        self.down1 = Conv(1, 48, 5 ,2 ,2)
        self.flat1 = Conv(48, 128, 3 ,1 ,1)
        self.flat2 = Conv(128, 128, 3 ,1 ,1)
        self.down2 = Conv(128, 128, 3, 2, 1)
        self.flat3 = Conv(128, 256, 3, 1, 1)
        self.flat4 = Conv(256, 256, 3, 1, 1)
        self.down3 = Conv(256, 256, 3, 2, 1)
        self.flat5 = Conv(256, 512, 3, 1, 1)
        self.flat6 = Conv(512, 1024, 3, 1, 1)
        self.flat7 = Conv(1024, 1024, 3, 1, 1)
        self.flat8 = Conv(1024, 1024, 3, 1, 1)
        self.flat9 = Conv(1024, 1024, 3, 1, 1)
        self.flat10 = Conv(1024, 512, 3, 1, 1)
        self.flat11 = Conv(512, 256, 3, 1, 1)
          
    def forward(self, x):
        x0 = self.down1(x)  
        x1 = self.flat1(x0)
        x2 = self.flat2(x1)
        x3 = self.down2(x2)
        x4 = self.flat3(x3)
        x5 = self.flat4(x4)
        x6 = self.down3(x5)
        x7 = self.flat5(x6)
        x8 = self.flat6(x7)
        x9 = self.flat7(x8)
        x10 = self.flat8(x9)
        x11 = self.flat9(x10)
        x12 = self.flat10(x11)
        x13 = self.flat11(x12)
        return x13

class Decoder(Module):
    def __init__(self):       
        super(Decoder, self).__init__()
        
        self.up1 = TransConv(256, 256, 4, 2, 1)
        self.flat1 = Conv(256, 256, 3, 1, 1)
        self.flat2 = Conv(256, 128, 3, 1, 1)
        self.up2 = TransConv(128, 128, 4, 2, 1)
        self.flat3 = Conv(128, 128, 3, 1, 1)
        self.flat4 = Conv(128, 48, 3, 1, 1)
        self.up3 = TransConv(48, 48, 4, 2, 1)
        self.flat5 = Conv(48, 24, 3, 1, 1)
        
        # layer cuoi cung Conv2d
        self.flat6 = nn.Conv2d(24, 1, 3, 1, 1)
          
    def forward(self, x):
        x0 = self.up1(x)  
        x1 = self.flat1(x0)
        x2 = self.flat2(x1)
        x3 = self.up2(x2)
        x4 = self.flat3(x3)
        x5 = self.flat4(x4)
        x6 = self.up3(x5)
        x7 = self.flat5(x6) 
        
        x8 = self.flat6(x7)
        
        return x8
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #layer cuoi dung tanh
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,alpha=0.2):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, 2, 2)
        self.leakyrelu1 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.leakyrelu2 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.leakyrelu3 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.leakyrelu4 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.leakyrelu5 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm5 = nn.BatchNorm2d(256)
        
        self.drop1 = nn.Dropout(0.5)
        
        self.conv6 = nn.Conv2d(256, 512, 3, 2, 1)
        self.leakyrelu6 = nn.LeakyReLU(alpha, inplace=True)
        self.batchNorm6 = nn.BatchNorm2d(512)
        
        self.drop2 = nn.Dropout(0.5)        
        self.linear = nn.Linear(512*4*4,1)   # 512*4*4  
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.batchNorm1(x) 
        
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.batchNorm2(x)
        
        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.batchNorm3(x)
        
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.batchNorm4(x)
        
        x = self.conv5(x)
        x = self.leakyrelu5(x)
        x = self.batchNorm5(x) 

        x = self.drop1(x)
    
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.batchNorm6(x)
        
        x = self.drop2(x)
        
        x = flatten(x, 1)        
        x = self.linear(x)        
        x = self.sigmoid(x)
        return x