import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class Conv(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
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
        self.relu = nn.ReLU()
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
        
        # layer cuoi cung Conv2d (khong ReLu)
        self.flat6 = nn.Conv2d(24, 1, 3, 1, 1)
        
        #self.bn6 = nn.BatchNorm2d(1)
          
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
        #x8 = self.bn6(x8)
        
        return x8
    
class ThinningNet(nn.Module):
    def __init__(self):
        super(ThinningNet, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn