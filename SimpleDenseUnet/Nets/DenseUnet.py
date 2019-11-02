import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import imageio
import numpy as np


class denselayer(nn.Module):
    def __init__(self,num_input_features,growth_rate,bn_size,drop_rate=0):
        super(denselayer,self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features,out_channels=bn_size*growth_rate,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
        )
        self.dropout = drop_rate
    def forward(self, input):
        new_features = self.layer(input)
        if self.dropout>0:
            new_features = F.dropout(new_features,p=self.dropout)
        output = torch.cat([input,new_features],1)
        #print("input shape:",input.shape,"; new_feature shape:",new_features.shape,"; output shape:",output.shape)
        return output

class denseblock(nn.Module):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate=0.0):
        super(denseblock,self).__init__()
        self.dense = nn.Sequential()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = denselayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.dense.add_module("denselayer%d"%(i+1),module=layer)
    def forward(self, input):
        output = self.dense(input)
        return output

class transition(nn.Module):
    def __init__(self,num_input_features,num_output_features):
        super(transition,self).__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features,num_output_features,kernel_size=1,stride=1,bias=False),
            nn.AvgPool2d(2,stride=2)
        )
    def forward(self, input):
        output = self.trans(input)
        return output

class up(nn.Module):
    def __init__(self,in_filters,n_filters):
        super(up,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU())
    def forward(self, x):
        x=self.deconv1(x)
        return x

class DenseNet2d(nn.Module):
    # initializers
    def __init__(self,num_classes=3, num_init_features=64,bn_size=2,growth_rate=16):
        super(DenseNet2d, self).__init__()
        self.num_classes = num_classes
        self.bn_size=bn_size

        # first conv2d
        self.inputtrans = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=num_init_features,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        # 如果要改模型的话，可以从这个conv2d的步长和pooling入手

        # dense block 1
        self.dense1 = denseblock(num_layers=4,num_input_features=num_init_features,bn_size=self.bn_size,growth_rate=16)
        input_features1 = num_init_features+4*growth_rate
        output_features1 = int(input_features1)
        self.trans1 = transition(num_input_features=input_features1,num_output_features=output_features1)

        # dense block 2
        growth_rate2 = 32
        self.dense2 = denseblock(num_layers=4,num_input_features=output_features1,bn_size=self.bn_size,growth_rate=growth_rate2)
        input_features2 = output_features1+4*growth_rate2
        output_features2 = int(input_features2)
        self.trans2 = transition(num_input_features=input_features2,num_output_features=output_features2)

        # dense block 3
        growth_rate3 = 32
        self.dense3 = denseblock(num_layers=4,num_input_features=output_features2,bn_size=self.bn_size,growth_rate=growth_rate3)
        input_features3 = output_features2+4*growth_rate3
        output_features3 = int(input_features3)
        self.trans3 = transition(num_input_features=input_features3,num_output_features=output_features3)

        # dense block 4
        growth_rate4 = 32
        self.dense4 = denseblock(num_layers=4,num_input_features=output_features3,bn_size=self.bn_size,growth_rate=growth_rate4)
        input_features4 = output_features3+4*growth_rate4
        output_features4 = int(input_features4)
        self.trans4 = transition(num_input_features=input_features4,num_output_features=output_features4)

        self.convmiddle = nn.Conv2d(in_channels=output_features4,out_channels=output_features4, kernel_size=3, stride=1, padding=1)

        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=(2,2))
        self.upsample = nn.Upsample(scale_factor=2)
        self.up1 = up(in_filters=2*output_features4,n_filters=output_features3)
        self.up2 = up(in_filters=2*output_features3,n_filters=output_features2)
        self.up3 = up(in_filters=2*output_features2,n_filters=output_features1)
        self.up4 = up(in_filters=2*output_features1,n_filters=num_init_features)
        self.up5 = up(in_filters=2*num_init_features,n_filters=32)

        self.convend1 = nn.Conv2d(in_channels=32,out_channels=self.num_classes,kernel_size=3,padding=1)
        self.convfinal = nn.Conv2d(in_channels=self.num_classes,out_channels=self.num_classes,kernel_size=1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        #print("init x shape:",x.size())
        #print("init x shape:",x.shape)
        x = self.inputtrans(x)
        #print("inconv shape:",x.shape)
        x1 = self.dense1(x)
        #print("dense1 shape:",x1.shape)
        x1 = self.trans1(x1)
        #print("x1 shape:",x1.shape)
        x2 = self.dense2(x1)
        #print("dense2 shape:",x2.shape)
        x2 = self.trans2(x2)
        #print("x2 shape:",x2.shape)
        x3 = self.dense3(x2)
        #print("dense3 shape:",x3.shape)
        x3 = self.trans3(x3)
        #print("x3 shape:",x3.shape)
        x4 = self.dense4(x3)
        #print("dense4 shape",x4.shape)
        x4 = self.trans4(x4)
        #print("x4 shape:",x4.shape)
        #x5 = self.dense5(x4)
        #print(x4.shape)
        #x5 = self.trans5(x5)
        #print("x4 shape:",x5.shape)

        xmiddle = self.convmiddle(x4)
        #print("xmiddle shape:",xmiddle.shape)

        x_up1 = self.up1(self.upsample(torch.cat([x4,xmiddle],dim=1)))
        #print("x_up1 shape:",x_up1.shape)
        x_up2 = self.up2(self.upsample(torch.cat([x_up1,x3],dim=1)))
        #print("x_up2 shape:",x_up2.shape)
        x_up3 = self.up3(self.upsample(torch.cat([x_up2,x2],dim=1)))
        #print("x_up3 shape:",x_up3.shape)
        x_up4 = self.up4(self.upsample(torch.cat([x_up3,x1],dim=1)))
        #print("x_up4 shape:",x_up4.shape)

        x_up5 = self.up5(self.upsample(torch.cat([x_up4,x],dim=1)))
        #print("x_up5 shape:",x_up5.shape)

        x_up6 = self.upsample(x_up5)
        #print("x_up6 shape:",x_up6.shape)
        x_final = self.convend1(x_up6)
        x_final = self.convfinal(x_final)
        #print("x final shape:",x_final.shape)
        #output = self.softmax(x_final)

        return x_final



if __name__ == "__main__":
    # model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,24,16))
    # print(model)
    b= DenseNet2d(num_classes=3,num_init_features=64,bn_size=4,growth_rate=16,)
    # #print(b)
    #
    te = imageio.imread('1.png')
    arr = np.expand_dims(te,axis=0)
    arr = np.expand_dims(arr,axis=0)
    arr = torch.FloatTensor(arr)
    cc = b(arr)

    pass
