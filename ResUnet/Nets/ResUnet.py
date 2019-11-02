import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import imageio

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation


        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias,0)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        #print("out1:",out1.shape)
        out2 = self.activation(self.bn1(self.conv2(out1)))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output

class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)

        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.bnup(self.up(x)))
        #crop1 = self.center_crop(bridge, up.size()[2])
        #print 'up.shape: ',up.shape, ' crop1.shape: ',crop1.shape
        crop1 = bridge
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out

class ResUNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=4):
        super(ResUNet, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        self.conv_block512_1024 = residualUnit(256, 512)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpResBlock(512,256)
        self.up_block512_256 = UNetUpResBlock(256,128)
        self.up_block256_128 = UNetUpResBlock(128,64)
        self.up_block128_64 = UNetUpResBlock(64,32)

        #self.conv64_64 = nn.Conv2d(32,32,3,1)
        self.last = nn.Conv2d(32, n_classes, 1, stride=1)

    def forward(self, x):
        #print("x shape:",x.shape)
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        #print("block1:",block1.shape)
        pool1 = self.pool1(block1)
        #print("pool1:",pool1.shape)

        block2 = self.conv_block64_128(pool1)
        #print("block2:",block2.shape)
        pool2 = self.pool2(block2)
        #print("pool2:",pool2.shape)

        block3 = self.conv_block128_256(pool2)
        #print("block3:",block3.shape)
        pool3 = self.pool3(block3)
        #print("pool3:",pool3.shape)

        block4 = self.conv_block256_512(pool3)
        #print("block4:",block4.shape)
        pool4 = self.pool4(block4)
        #print("pool4:",pool4.shape)
        #
        block5 = self.conv_block512_1024(pool4)
        #print("block5:",block5.shape)
        #
        up1 = self.up_block1024_512(block5, block4)
        #print("up1:",up1.shape)

        up2 = self.up_block512_256(up1, block3)
        #print("up2:",up2.shape)

        up3 = self.up_block256_128(up2, block2)
        #print("up3:",up3.shape)

        up4 = self.up_block128_64(up3, block1)
        #print("up4:",up4.shape)
        #up4 = self.conv64_64(up4)
        output = self.last(up4)
        #print("output:",output.shape)

        return output



if __name__ == "__main__":
    # model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,24,16))
    # print(model)
    b= ResUNet(in_channel=1,n_classes=3)
    # #print(b)
    #
    te = imageio.imread('1.png')
    arr = np.expand_dims(te,axis=0)
    arr = np.expand_dims(arr,axis=0)
    arr = torch.FloatTensor(arr)
    cc = b(arr)

    pass
