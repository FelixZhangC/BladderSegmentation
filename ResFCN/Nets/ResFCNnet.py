import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
import imageio

from Nets.nnBuildUnits import *

class ResSegNet(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(ResSegNet, self).__init__()
        #self.imsize = imsize

        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'

        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation

        self.activation = F.relu

        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)

        self.conv_block1_64 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block64_64 = residualUnit3(32, 32, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block64_128 = residualUnit3(32, 64, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block128_256 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)

        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)

        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0.05, nd=nd)

            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0, nd=nd)

        self.last = conv23DUnit(32, out_channels, 1, nd=nd)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        #print("x shape：",x.shape)
        block0 = self.conv_block1_64(x)
        #print("block0 shape:",block0.shape)
        block1 = self.conv_block64_64(block0)
        #print("block1 shape:",block1.shape)
        pool1 = self.pool1(block1)
        #print("pool1 shape:",pool1.shape)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block64_128(pool1)
        #print("block2 shape:",block2.shape)
        pool2 = self.pool2(block2)
        #print("pool2 shape:",pool2.shape)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block128_256(pool2)
        #print("block3 shape:",block3.shape)
        pool3 = self.pool3(block3)
        #print("pool3 shape:",pool3.shape)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block256_512(pool3)
        #print("block4 shape:",block4.shape)


        up2 = self.up_block512_256(block4, block3)
        #print("up2 shape:",up2.shape)
        up3 = self.up_block256_128(up2, block2)
        #print("up3 shape:",up3.shape)
        up4 = self.up_block128_64(up3, block1)
        #print("up4 shape:",up4.shape)

        output = self.last(up4)
        #print("last shape:",output.shape)
        output = self.logsoftmax(output)
        #print("output shape:",output.shape)
#         return F.log_softmax(self.last(up4))
        return output


if __name__ == "__main__":
    m = ResSegNet(in_channels=1,out_channels=3)
    a = imageio.imread("1.png")
    a = np.expand_dims(a,axis=0)
    a = np.expand_dims(a,axis=0)
    aa = torch.Tensor(a)
    b = m(aa)
    print(b.shape)
    pass
