import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import imageio
import numpy as np

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,nb_filter):
        super(conv_block,self).__init__()
        self.batchnorm1 = nn.BatchNorm2d(num_features=in_channels)
        self.activation1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=nb_filter*4,kernel_size=1,padding=0,bias=False)

        self.batchnorm2 = nn.BatchNorm2d(num_features=nb_filter*4)
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=nb_filter*4,out_channels=out_channels,kernel_size=3,padding=1,bias=False)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.conv2(x)

        return x

class dense_block1(nn.Module):
    def __init__(self,input_size,output_size,growth_rate):
        super(dense_block1,self).__init__()
        self.layer1 = conv_block(in_channels=input_size,out_channels=growth_rate,nb_filter=64)
        self.layer2 = conv_block(in_channels=input_size+growth_rate,out_channels=growth_rate,nb_filter=64)
        self.layer3 = conv_block(in_channels=input_size+growth_rate*2,out_channels=growth_rate,nb_filter=64)
        self.layer4 = conv_block(in_channels=input_size+growth_rate*3,out_channels=growth_rate,nb_filter=64)
        self.layer5 = conv_block(in_channels=input_size+growth_rate*4,out_channels=growth_rate,nb_filter=64)
        self.layer6 = conv_block(in_channels=input_size+growth_rate*5,out_channels=growth_rate,nb_filter=64)

    def forward(self, x):
        #print("x shape:",x.shape)
        x1 = self.layer1(x)
        #print("x1 shape:",x1.shape)
        layer1cat = torch.cat([x1,x],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x2 = self.layer2(layer1cat)
        layer2cat = torch.cat([x2,layer1cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x3 = self.layer3(layer2cat)
        layer3cat = torch.cat([x3,layer2cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x4 = self.layer4(layer3cat)
        layer4cat = torch.cat([x4,layer3cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x5 = self.layer5(layer4cat)
        layer5cat = torch.cat([x5,layer4cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x6 = self.layer6(layer5cat)
        layer6cat = torch.cat([x6,layer5cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        return layer6cat

class dense_block2(nn.Module):
    def __init__(self,input_size,output_size,growth_rate):
        super(dense_block2,self).__init__()
        self.layer1 = conv_block(in_channels=input_size,out_channels=growth_rate,nb_filter=64)
        self.layer2 = conv_block(in_channels=input_size+growth_rate,out_channels=growth_rate,nb_filter=64)
        self.layer3 = conv_block(in_channels=input_size+growth_rate*2,out_channels=growth_rate,nb_filter=64)
        self.layer4 = conv_block(in_channels=input_size+growth_rate*3,out_channels=growth_rate,nb_filter=64)
        self.layer5 = conv_block(in_channels=input_size+growth_rate*4,out_channels=growth_rate,nb_filter=64)
        self.layer6 = conv_block(in_channels=input_size+growth_rate*5,out_channels=growth_rate,nb_filter=64)
        self.layer7 = conv_block(in_channels=input_size+growth_rate*6,out_channels=growth_rate,nb_filter=64)
        self.layer8 = conv_block(in_channels=input_size+growth_rate*7,out_channels=growth_rate,nb_filter=64)
        self.layer9 = conv_block(in_channels=input_size+growth_rate*8,out_channels=growth_rate,nb_filter=64)
        self.layer10 = conv_block(in_channels=input_size+growth_rate*9,out_channels=growth_rate,nb_filter=64)
        self.layer11 = conv_block(in_channels=input_size+growth_rate*10,out_channels=growth_rate,nb_filter=64)
        self.layer12 = conv_block(in_channels=input_size+growth_rate*11,out_channels=growth_rate,nb_filter=64)
    def forward(self, x):
        #print("x shape:",x.shape)
        x1 = self.layer1(x)
        #print("x1 shape:",x1.shape)
        layer1cat = torch.cat([x1,x],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x2 = self.layer2(layer1cat)
        layer2cat = torch.cat([x2,layer1cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x3 = self.layer3(layer2cat)
        layer3cat = torch.cat([x3,layer2cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x4 = self.layer4(layer3cat)
        layer4cat = torch.cat([x4,layer3cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x5 = self.layer5(layer4cat)
        layer5cat = torch.cat([x5,layer4cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x6 = self.layer6(layer5cat)
        layer6cat = torch.cat([x6,layer5cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x7 = self.layer7(layer6cat)
        layer7cat = torch.cat([x7,layer6cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x8 = self.layer8(layer7cat)
        layer8cat = torch.cat([x8,layer7cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x9 = self.layer9(layer8cat)
        layer9cat = torch.cat([x9,layer8cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x10 = self.layer10(layer9cat)
        layer10cat = torch.cat([x10,layer9cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x11 = self.layer11(layer10cat)
        layer11cat = torch.cat([x11,layer10cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x12 = self.layer12(layer11cat)
        layer12cat = torch.cat([x12,layer11cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        return layer12cat

class dense_block3(nn.Module):
    def __init__(self,input_size,output_size,growth_rate):
        super(dense_block3,self).__init__()
        self.layer1 = conv_block(in_channels=input_size,out_channels=growth_rate,nb_filter=64)
        self.layer2 = conv_block(in_channels=input_size+growth_rate,out_channels=growth_rate,nb_filter=64)
        self.layer3 = conv_block(in_channels=input_size+growth_rate*2,out_channels=growth_rate,nb_filter=64)
        self.layer4 = conv_block(in_channels=input_size+growth_rate*3,out_channels=growth_rate,nb_filter=64)
        self.layer5 = conv_block(in_channels=input_size+growth_rate*4,out_channels=growth_rate,nb_filter=64)
        self.layer6 = conv_block(in_channels=input_size+growth_rate*5,out_channels=growth_rate,nb_filter=64)
        self.layer7 = conv_block(in_channels=input_size+growth_rate*6,out_channels=growth_rate,nb_filter=64)
        self.layer8 = conv_block(in_channels=input_size+growth_rate*7,out_channels=growth_rate,nb_filter=64)
        self.layer9 = conv_block(in_channels=input_size+growth_rate*8,out_channels=growth_rate,nb_filter=64)
        self.layer10 = conv_block(in_channels=input_size+growth_rate*9,out_channels=growth_rate,nb_filter=64)

        self.layer11 = conv_block(in_channels=input_size+growth_rate*10,out_channels=growth_rate,nb_filter=64)
        self.layer12 = conv_block(in_channels=input_size+growth_rate*11,out_channels=growth_rate,nb_filter=64)
        self.layer13 = conv_block(in_channels=input_size+growth_rate*12,out_channels=growth_rate,nb_filter=64)
        self.layer14 = conv_block(in_channels=input_size+growth_rate*13,out_channels=growth_rate,nb_filter=64)
        self.layer15 = conv_block(in_channels=input_size+growth_rate*14,out_channels=growth_rate,nb_filter=64)
        self.layer16 = conv_block(in_channels=input_size+growth_rate*15,out_channels=growth_rate,nb_filter=64)
        self.layer17 = conv_block(in_channels=input_size+growth_rate*16,out_channels=growth_rate,nb_filter=64)
        self.layer18 = conv_block(in_channels=input_size+growth_rate*17,out_channels=growth_rate,nb_filter=64)
        self.layer19 = conv_block(in_channels=input_size+growth_rate*18,out_channels=growth_rate,nb_filter=64)
        self.layer20 = conv_block(in_channels=input_size+growth_rate*19,out_channels=growth_rate,nb_filter=64)

        self.layer21 = conv_block(in_channels=input_size+growth_rate*20,out_channels=growth_rate,nb_filter=64)
        self.layer22 = conv_block(in_channels=input_size+growth_rate*21,out_channels=growth_rate,nb_filter=64)
        self.layer23 = conv_block(in_channels=input_size+growth_rate*22,out_channels=growth_rate,nb_filter=64)
        self.layer24 = conv_block(in_channels=input_size+growth_rate*23,out_channels=growth_rate,nb_filter=64)
        self.layer25 = conv_block(in_channels=input_size+growth_rate*24,out_channels=growth_rate,nb_filter=64)
        self.layer26 = conv_block(in_channels=input_size+growth_rate*25,out_channels=growth_rate,nb_filter=64)
        self.layer27 = conv_block(in_channels=input_size+growth_rate*26,out_channels=growth_rate,nb_filter=64)
        self.layer28 = conv_block(in_channels=input_size+growth_rate*27,out_channels=growth_rate,nb_filter=64)
        self.layer29 = conv_block(in_channels=input_size+growth_rate*28,out_channels=growth_rate,nb_filter=64)
        self.layer30 = conv_block(in_channels=input_size+growth_rate*29,out_channels=growth_rate,nb_filter=64)

        self.layer31 = conv_block(in_channels=input_size+growth_rate*30,out_channels=growth_rate,nb_filter=64)
        self.layer32 = conv_block(in_channels=input_size+growth_rate*31,out_channels=growth_rate,nb_filter=64)
        self.layer33 = conv_block(in_channels=input_size+growth_rate*32,out_channels=growth_rate,nb_filter=64)
        self.layer34 = conv_block(in_channels=input_size+growth_rate*33,out_channels=growth_rate,nb_filter=64)
        self.layer35 = conv_block(in_channels=input_size+growth_rate*34,out_channels=growth_rate,nb_filter=64)
        self.layer36 = conv_block(in_channels=input_size+growth_rate*35,out_channels=growth_rate,nb_filter=64)


    def forward(self, x):
        #print("x shape:",x.shape)
        x1 = self.layer1(x)
        #print("x1 shape:",x1.shape)
        layer1cat = torch.cat([x1,x],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x2 = self.layer2(layer1cat)
        layer2cat = torch.cat([x2,layer1cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x3 = self.layer3(layer2cat)
        layer3cat = torch.cat([x3,layer2cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x4 = self.layer4(layer3cat)
        layer4cat = torch.cat([x4,layer3cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x5 = self.layer5(layer4cat)
        layer5cat = torch.cat([x5,layer4cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x6 = self.layer6(layer5cat)
        layer6cat = torch.cat([x6,layer5cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x7 = self.layer7(layer6cat)
        layer7cat = torch.cat([x7,layer6cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x8 = self.layer8(layer7cat)
        layer8cat = torch.cat([x8,layer7cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x9 = self.layer9(layer8cat)
        layer9cat = torch.cat([x9,layer8cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x10 = self.layer10(layer9cat)
        layer10cat = torch.cat([x10,layer9cat],dim=1)
        #print("layercat10 shape:",layer10cat.shape)

        x11 = self.layer11(layer10cat)
        #print("x1 shape:",x1.shape)
        layer11cat = torch.cat([x11,layer10cat],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x12 = self.layer12(layer11cat)
        layer12cat = torch.cat([x12,layer11cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x13 = self.layer13(layer12cat)
        layer13cat = torch.cat([x13,layer12cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x14 = self.layer14(layer13cat)
        layer14cat = torch.cat([x14,layer13cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x15 = self.layer15(layer14cat)
        layer15cat = torch.cat([x15,layer14cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x16 = self.layer16(layer15cat)
        layer16cat = torch.cat([x16,layer15cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x17 = self.layer17(layer16cat)
        layer17cat = torch.cat([x17,layer16cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x18 = self.layer18(layer17cat)
        layer18cat = torch.cat([x18,layer17cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x19 = self.layer19(layer18cat)
        layer19cat = torch.cat([x19,layer18cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x20 = self.layer20(layer19cat)
        layer20cat = torch.cat([x20,layer19cat],dim=1)
        #print("layercat10 shape:",layer10cat.shape)

        x21 = self.layer21(layer20cat)
        #print("x1 shape:",x1.shape)
        layer21cat = torch.cat([x21,layer20cat],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x22 = self.layer22(layer21cat)
        layer22cat = torch.cat([x22,layer21cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x23 = self.layer23(layer22cat)
        layer23cat = torch.cat([x23,layer22cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x24 = self.layer24(layer23cat)
        layer24cat = torch.cat([x24,layer23cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x25 = self.layer25(layer24cat)
        layer25cat = torch.cat([x25,layer24cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x26 = self.layer26(layer25cat)
        layer26cat = torch.cat([x26,layer25cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x27 = self.layer27(layer26cat)
        layer27cat = torch.cat([x27,layer26cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x28 = self.layer28(layer27cat)
        layer28cat = torch.cat([x28,layer27cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x29 = self.layer29(layer28cat)
        layer29cat = torch.cat([x29,layer28cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x30 = self.layer30(layer29cat)
        layer30cat = torch.cat([x30,layer29cat],dim=1)
        #print("layercat10 shape:",layer10cat.shape)

        x31 = self.layer31(layer30cat)
        #print("x1 shape:",x1.shape)
        layer31cat = torch.cat([x31,layer30cat],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x32 = self.layer32(layer31cat)
        layer32cat = torch.cat([x32,layer31cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x33 = self.layer33(layer32cat)
        layer33cat = torch.cat([x33,layer32cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x34 = self.layer34(layer33cat)
        layer34cat = torch.cat([x34,layer33cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x35 = self.layer35(layer34cat)
        layer35cat = torch.cat([x35,layer34cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x36 = self.layer36(layer35cat)
        layer36cat = torch.cat([x36,layer35cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        return layer36cat

class dense_block4(nn.Module):
    def __init__(self,input_size,output_size,growth_rate):
        super(dense_block4,self).__init__()
        self.layer1 = conv_block(in_channels=input_size,out_channels=growth_rate,nb_filter=64)
        self.layer2 = conv_block(in_channels=input_size+growth_rate,out_channels=growth_rate,nb_filter=64)
        self.layer3 = conv_block(in_channels=input_size+growth_rate*2,out_channels=growth_rate,nb_filter=64)
        self.layer4 = conv_block(in_channels=input_size+growth_rate*3,out_channels=growth_rate,nb_filter=64)
        self.layer5 = conv_block(in_channels=input_size+growth_rate*4,out_channels=growth_rate,nb_filter=64)
        self.layer6 = conv_block(in_channels=input_size+growth_rate*5,out_channels=growth_rate,nb_filter=64)
        self.layer7 = conv_block(in_channels=input_size+growth_rate*6,out_channels=growth_rate,nb_filter=64)
        self.layer8 = conv_block(in_channels=input_size+growth_rate*7,out_channels=growth_rate,nb_filter=64)
        self.layer9 = conv_block(in_channels=input_size+growth_rate*8,out_channels=growth_rate,nb_filter=64)
        self.layer10 = conv_block(in_channels=input_size+growth_rate*9,out_channels=growth_rate,nb_filter=64)

        self.layer11 = conv_block(in_channels=input_size+growth_rate*10,out_channels=growth_rate,nb_filter=64)
        self.layer12 = conv_block(in_channels=input_size+growth_rate*11,out_channels=growth_rate,nb_filter=64)
        self.layer13 = conv_block(in_channels=input_size+growth_rate*12,out_channels=growth_rate,nb_filter=64)
        self.layer14 = conv_block(in_channels=input_size+growth_rate*13,out_channels=growth_rate,nb_filter=64)
        self.layer15 = conv_block(in_channels=input_size+growth_rate*14,out_channels=growth_rate,nb_filter=64)
        self.layer16 = conv_block(in_channels=input_size+growth_rate*15,out_channels=growth_rate,nb_filter=64)
        self.layer17 = conv_block(in_channels=input_size+growth_rate*16,out_channels=growth_rate,nb_filter=64)
        self.layer18 = conv_block(in_channels=input_size+growth_rate*17,out_channels=growth_rate,nb_filter=64)
        self.layer19 = conv_block(in_channels=input_size+growth_rate*18,out_channels=growth_rate,nb_filter=64)
        self.layer20 = conv_block(in_channels=input_size+growth_rate*19,out_channels=growth_rate,nb_filter=64)

        self.layer21 = conv_block(in_channels=input_size+growth_rate*20,out_channels=growth_rate,nb_filter=64)
        self.layer22 = conv_block(in_channels=input_size+growth_rate*21,out_channels=growth_rate,nb_filter=64)
        self.layer23 = conv_block(in_channels=input_size+growth_rate*22,out_channels=growth_rate,nb_filter=64)
        self.layer24 = conv_block(in_channels=input_size+growth_rate*23,out_channels=growth_rate,nb_filter=64)



    def forward(self, x):
        #print("x shape:",x.shape)
        x1 = self.layer1(x)
        #print("x1 shape:",x1.shape)
        layer1cat = torch.cat([x1,x],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x2 = self.layer2(layer1cat)
        layer2cat = torch.cat([x2,layer1cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x3 = self.layer3(layer2cat)
        layer3cat = torch.cat([x3,layer2cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x4 = self.layer4(layer3cat)
        layer4cat = torch.cat([x4,layer3cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x5 = self.layer5(layer4cat)
        layer5cat = torch.cat([x5,layer4cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x6 = self.layer6(layer5cat)
        layer6cat = torch.cat([x6,layer5cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x7 = self.layer7(layer6cat)
        layer7cat = torch.cat([x7,layer6cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x8 = self.layer8(layer7cat)
        layer8cat = torch.cat([x8,layer7cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x9 = self.layer9(layer8cat)
        layer9cat = torch.cat([x9,layer8cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x10 = self.layer10(layer9cat)
        layer10cat = torch.cat([x10,layer9cat],dim=1)
        #print("layercat10 shape:",layer10cat.shape)

        x11 = self.layer11(layer10cat)
        #print("x1 shape:",x1.shape)
        layer11cat = torch.cat([x11,layer10cat],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x12 = self.layer12(layer11cat)
        layer12cat = torch.cat([x12,layer11cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x13 = self.layer13(layer12cat)
        layer13cat = torch.cat([x13,layer12cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x14 = self.layer14(layer13cat)
        layer14cat = torch.cat([x14,layer13cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x15 = self.layer15(layer14cat)
        layer15cat = torch.cat([x15,layer14cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x16 = self.layer16(layer15cat)
        layer16cat = torch.cat([x16,layer15cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x17 = self.layer17(layer16cat)
        layer17cat = torch.cat([x17,layer16cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x18 = self.layer18(layer17cat)
        layer18cat = torch.cat([x18,layer17cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x19 = self.layer19(layer18cat)
        layer19cat = torch.cat([x19,layer18cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        x20 = self.layer20(layer19cat)
        layer20cat = torch.cat([x20,layer19cat],dim=1)
        #print("layercat10 shape:",layer10cat.shape)

        x21 = self.layer21(layer20cat)
        #print("x1 shape:",x1.shape)
        layer21cat = torch.cat([x21,layer20cat],dim=1)
        #print("layercat1 shape:",layer1cat.shape)

        x22 = self.layer22(layer21cat)
        layer22cat = torch.cat([x22,layer21cat],dim=1)
        #print("layercat2 shape:",layer2cat.shape)

        x23 = self.layer23(layer22cat)
        layer23cat = torch.cat([x23,layer22cat],dim=1)
        #print("layercat3 shape:",layer3cat.shape)

        x24 = self.layer24(layer23cat)
        layer24cat = torch.cat([x24,layer23cat],dim=1)
        #print("layercat4 shape:",layer4cat.shape)

        return layer24cat

# class dense_block4(nn.Module):
#     def __init__(self,input_size,output_size,growth_rate):
#         super(dense_block4,self).__init__()
#         self.layer1 = conv_block(in_channels=input_size,out_channels=growth_rate,nb_filter=16)
#         self.layer2 = conv_block(in_channels=input_size+growth_rate,out_channels=growth_rate,nb_filter=16)
#         self.layer3 = conv_block(in_channels=input_size+growth_rate*2,out_channels=growth_rate,nb_filter=16)
#         self.layer4 = conv_block(in_channels=input_size+growth_rate*3,out_channels=growth_rate,nb_filter=16)
#         self.layer5 = conv_block(in_channels=input_size+growth_rate*4,out_channels=growth_rate,nb_filter=16)
#         self.layer6 = conv_block(in_channels=input_size+growth_rate*5,out_channels=growth_rate,nb_filter=16)
#         self.layer7 = conv_block(in_channels=input_size+growth_rate*6,out_channels=growth_rate,nb_filter=16)
#         self.layer8 = conv_block(in_channels=input_size+growth_rate*7,out_channels=growth_rate,nb_filter=16)
#         self.layer9 = conv_block(in_channels=input_size+growth_rate*8,out_channels=growth_rate,nb_filter=16)
#         self.layer10 = conv_block(in_channels=input_size+growth_rate*9,out_channels=growth_rate,nb_filter=16)
#
#     def forward(self, x):
#         #print("x shape:",x.shape)
#         x1 = self.layer1(x)
#         #print("x1 shape:",x1.shape)
#         layer1cat = torch.cat([x1,x],dim=1)
#         #print("layercat1 shape:",layer1cat.shape)
#
#         x2 = self.layer2(layer1cat)
#         layer2cat = torch.cat([x2,layer1cat],dim=1)
#         #print("layercat2 shape:",layer2cat.shape)
#
#         x3 = self.layer3(layer2cat)
#         layer3cat = torch.cat([x3,layer2cat],dim=1)
#         #print("layercat3 shape:",layer3cat.shape)
#
#         x4 = self.layer4(layer3cat)
#         layer4cat = torch.cat([x4,layer3cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x5 = self.layer5(layer4cat)
#         layer5cat = torch.cat([x5,layer4cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x6 = self.layer6(layer5cat)
#         layer6cat = torch.cat([x6,layer5cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x7 = self.layer7(layer6cat)
#         layer7cat = torch.cat([x7,layer6cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x8 = self.layer8(layer7cat)
#         layer8cat = torch.cat([x8,layer7cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x9 = self.layer9(layer8cat)
#         layer9cat = torch.cat([x9,layer8cat],dim=1)
#         #print("layercat4 shape:",layer4cat.shape)
#
#         x10 = self.layer10(layer9cat)
#         layer10cat = torch.cat([x10,layer9cat],dim=1)
#         #print("layercat10 shape:",layer10cat.shape)
#
#         return layer10cat

class transition_block(nn.Module):
    def __init__(self,input_size,output_size):
        super(transition_block,self).__init__()

        self.batchnorm = nn.BatchNorm2d(num_features=input_size)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=input_size,out_channels=output_size,kernel_size=1,bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseUNet(nn.Module):
    def __init__(self,in_size=1,num_classes=3,growth_rate=16,compression=0.5):
        super(DenseUNet,self).__init__()

        num_init_features = 96
        self.inputconv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=num_init_features,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )
        self.initpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        block1in_num = num_init_features
        block1out_num = num_init_features+6*48
        self.block1 = dense_block1(input_size=block1in_num,output_size=block1out_num,growth_rate=48)
        trans1out_num = int(block1out_num*compression)
        self.trans1 = transition_block(input_size=block1out_num,output_size=trans1out_num)

        block2in_num = trans1out_num
        block2out_num = trans1out_num+12*48
        self.block2 = dense_block2(input_size=block2in_num,output_size=block2out_num,growth_rate=48)
        trans2out_num = int(block2out_num*compression)
        self.trans2 = transition_block(input_size=block2out_num,output_size=trans2out_num)

        block3in_num = trans2out_num
        block3out_num = trans2out_num+36*48
        self.block3 = dense_block3(input_size=block3in_num,output_size=block3out_num,growth_rate=48)
        trans3out_num = int(block3out_num*compression)
        self.trans3 = transition_block(input_size=block3out_num,output_size=trans3out_num)

        block4in_num = trans3out_num
        block4out_num = trans3out_num+24*48
        self.block4 = dense_block4(input_size=block4in_num,output_size=block4out_num,growth_rate=48)
        #trans4out_num = int(block4out_num*compression)
        #self.trans4 = transition_block(input_size=block4out_num,output_size=trans4out_num)

        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()

        self.line0conv = nn.Conv2d(in_channels=block3out_num,out_channels=block4out_num,kernel_size=1)

        self.up0conv = nn.Conv2d(in_channels=block4out_num,out_channels=block2out_num,kernel_size=3,stride=1,padding=1)
        self.up0batchnorm = nn.BatchNorm2d(num_features=block2out_num)

        self.up1conv = nn.Conv2d(in_channels=block2out_num,out_channels=block1out_num,kernel_size=3,stride=1,padding=1)
        self.up1batchnorm = nn.BatchNorm2d(num_features=block1out_num)

        self.up2conv = nn.Conv2d(in_channels=block1out_num,out_channels=num_init_features,kernel_size=3,stride=1,padding=1)
        self.up2batchnorm = nn.BatchNorm2d(num_features=num_init_features)

        self.up3conv = nn.Conv2d(in_channels=num_init_features,out_channels=num_init_features,kernel_size=3,stride=1,padding=1)
        self.up3batchnorm = nn.BatchNorm2d(num_features=num_init_features)

        self.up4conv = nn.Conv2d(in_channels=num_init_features,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.up4batchnorm = nn.BatchNorm2d(num_features=64)

        self.finalconv = nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1,stride=1,padding=0)





    def forward(self, input):
        #print("input shape:",input.shape)
        x = self.inputconv(input)
        #print("x shape:",x.shape)
        x_pool = self.initpool(x)
        #print("init pool x shape:",x_pool.shape)
        #print()

        block1 = self.block1(x_pool)
        #print("block1 shape:",block1.shape)
        trans1 = self.trans1(block1)
        #print("trans1 shape:",trans1.shape)
        #print()

        block2 = self.block2(trans1)
        #print("block2 shape:",block2.shape)
        trans2 = self.trans2(block2)
        #print("trans2 shape:",trans2.shape)
        #print()

        block3 = self.block3(trans2)
        #print("block3 shape:",block3.shape)
        trans3 = self.trans3(block3)
        #print("trans3 shape:",trans3.shape)
        #print()

        block4 = self.block4(trans3)
        #print("block4 shape:",block4.shape)
        #trans4 = self.trans4(block4)
        #print("trans4 shape:",trans4.shape)


        #print()
        up0 = self.upsample(block4)
        box3 = self.line0conv(block3)
        #print("up0 shape:",up0.shape)
        #print("box3 shape:",box3.shape)
        up0sum = torch.add(box3,up0)
        #print("box3sum shape:",up0sum.shape)
        conv_up0 = self.up0conv(up0sum)
        bn_up0 = self.up0batchnorm(conv_up0)
        ac_up0 = self.relu(bn_up0)
        #print("ac_up0 shape:",ac_up0.shape)
        #print()

        up1= self.upsample(ac_up0)
        #print("up1 shape:",up1.shape)
        up1sum = torch.add(block2,up1)
        #print("up1sum shape:",up1sum.shape)
        conv_up1 = self.up1conv(up1sum)
        bn_up1 = self.up1batchnorm(conv_up1)
        ac_up1 = self.relu(bn_up1)
        #print("ac_up1 shape:",ac_up1.shape)
        #print()

        up2= self.upsample(ac_up1)
        #print("up2 shape:",up2.shape)
        up2sum = torch.add(block1,up2)
        #print("up2sum shape:",up2sum.shape)
        conv_up2 = self.up2conv(up2sum)
        bn_up2 = self.up2batchnorm(conv_up2)
        ac_up2 = self.relu(bn_up2)
        #print("ac_up2 shape:",ac_up2.shape)
        #print()

        up3= self.upsample(ac_up2)
        #print("up3 shape:",up3.shape)
        up3sum = torch.add(x,up3)
        #print("up3sum shape:",up3sum.shape)
        conv_up3 = self.up3conv(up3sum)
        bn_up3 = self.up3batchnorm(conv_up3)
        ac_up3 = self.relu(bn_up3)
        #print("ac_up2 shape:",ac_up3.shape)
        #print()

        up4 = self.upsample(ac_up3)
        #print("up4 shape:",up4.shape)
        up4conv = self.up4conv(up4)
        bn_up4 = self.up4batchnorm(up4conv)
        ac_up4 = self.relu(bn_up4)
        #print("ac_up4 shape:",ac_up4.shape)

        final = self.finalconv(ac_up4)
        #print("final shape:",final.shape)
        return final

if __name__ == "__main__":
    # model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,24,16))
    # print(model)
    b= DenseUNet(growth_rate=16)
    # #print(b)

    te = imageio.imread('1.png')
    arr = np.expand_dims(te,axis=0)
    arr = np.expand_dims(arr,axis=0)
    arr = torch.FloatTensor(arr)
    cc = b(arr)

    pass
