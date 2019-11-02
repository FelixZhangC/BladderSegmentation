import torch
from torch.utils import data
from torch.optim import Adam
from torchvision.transforms import Compose
import time
import os
import numpy as np
import imageio

from datasets import BladderOneHot
from Nets.ResFCNnet import ResSegNet

from loss import Cross_Entropy2d,myWeightedDiceLoss4Organs
from metric import Dice
from utils import ToFloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train():

    # settings
    learning_rate = 1e-4
    total_epoches = 4000
    batchsize = 4
    device = "cuda"

    # data transform
    input_transform = Compose([ToFloatTensor(),])
    target_transform = Compose([ToFloatTensor(),])

    # dataloader
    trainloader = data.DataLoader(BladderOneHot(imageroot="../../GithubData/Image/",labelroot="../../GithubData/Label/",img_transform=input_transform,label_transform=target_transform),batch_size=batchsize,shuffle=True)
    valloader = data.DataLoader(BladderOneHot(imageroot="../../GithubData/Image/",labelroot="../../GithubData/Label/",img_transform=input_transform,label_transform=target_transform),batch_size=batchsize,shuffle=True)
    #valloader = None
    # model
    #G = Unet(n_classes=2,n_filters=32).to(device)
    G = ResSegNet(in_channels=1,out_channels=3).to(device)

    # optimizer
    optimizer_G = Adam(G.parameters(),lr=learning_rate,betas=(0.5,0.9),eps=10e-8)

    # train
    for epoch in range(total_epoches):
        G.train()

        dice_sum = 0.0
        batch_num = 0
        for real_imgs,real_labels in trainloader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            #print("real ima shape:",real_imgs.shape)
            #print("real labels shape:",real_labels.shape)
            G.zero_grad()
            optimizer_G.zero_grad()

            pred_labels = G(real_imgs)
            #seg_loss = Cross_Entropy2d(pred_labels,real_labels)
            #seg_loss.backward(retain_graph=True)
            seg_loss = myWeightedDiceLoss4Organs(pred_labels,real_labels)
            seg_loss.backward()

            # calculate matric
            batch_num += 1
            dice = Dice(pred_labels,real_labels)
            dice_sum += dice

            optimizer_G.step()


        present_time = time.strftime("%Y.%M.%D.%H:%M:%S", time.localtime(time.time()))
        dice_per_epoch = dice_sum/batch_num
        print("epoch[%d/%d] segloss%f ; dice:%.4f ; Time: %s ;"%(epoch,total_epoches,seg_loss,dice_per_epoch,present_time))

        # eval
        G.eval()
        if epoch%5 == 0:
            if valloader != None:
                ave_dice = 0.0
                ave_val_loss = 0.0
                batch_num = 0
                for imgs,labels in valloader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    pred_val_labels = G(imgs)

                    dice = Dice(pred_val_labels,labels)
                    val_loss = Cross_Entropy2d(pred_val_labels,labels)

                    batch_num += 1
                    ave_dice += dice
                    ave_val_loss += val_loss.cpu().detach().numpy()
                ave_dice /= batch_num
                ave_val_loss /= batch_num
                print("Evalution on valset -- dice: %.4f ; val loss: %.4f;"%(ave_dice,ave_val_loss))
            # save model
            Gname = "./checkpoint/G_epoch_"+str(epoch)+".pth"
            torch.save(G.state_dict(),Gname)








if __name__=="__main__":
    present_time = time.strftime("%Y.%M.%D.%H:%M:%S", time.localtime(time.time()))
    print(present_time)
    train()
    pass
