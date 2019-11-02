import torch
from torch.utils import data
from torch.optim import Adam
from torchvision.transforms import Compose
import time
import os
import numpy as np
import imageio

from datasets import BladderOneHot
from Nets.DenseUnet import DenseNet2d
from loss import WeightedDiceLoss,WeightedCrossEntropy
from metric import Dice
from utils import ToFloatTensor


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    # settings
    learning_rate = 1e-4
    total_epoches = 4000
    batchsize = 8
    device = "cuda"
    modelsavepath = "./checkpoint/"

    # data transform
    input_transform = Compose([ToFloatTensor(),])
    target_transform = Compose([ToFloatTensor(),])
    # dataloader
    trainloader = data.DataLoader(BladderOneHot(imageroot="../../../0821/KingData/Image/",labelroot="../../../0821/KingData/Label/",img_transform=input_transform,label_transform=target_transform),batch_size=batchsize,shuffle=True)

    # model
    G = DenseNet2d(num_classes=3).to(device)

    # optimizer
    optimizer_G = Adam(G.parameters(),lr=learning_rate,betas=(0.5,0.9),eps=10e-8)

    # train
    best_val_loss = np.inf
    for epoch in range(total_epoches):
        G.train()
        batch_num = 0
        ave_train_loss = 0.0
        for real_imgs,real_labels in trainloader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            G.zero_grad()
            optimizer_G.zero_grad()

            pred_labels = G(real_imgs)
            #seg_loss = Cross_Entropy2d(pred_labels,real_labels)
            seg_loss = 0.5*WeightedDiceLoss(pred_labels,real_labels) + 0.5*WeightedCrossEntropy(pred_labels,real_labels)
            seg_loss.backward()

            # calculate matric
            batch_num += 1
            ave_train_loss += seg_loss.cpu().detach().numpy()
            optimizer_G.step()
        ave_train_loss /= batch_num


        # eval
        G.eval()
        if epoch %10 == 0:
            modelname = "G_epoch"+str(epoch)+".pth"
            Gname = os.path.join(modelsavepath,modelname)
            torch.save(G.state_dict(),Gname)
        present_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        print("epoch[%d/%d]  TrainDiceLoss :%.8f ; Time: %s"%(epoch,total_epoches,ave_train_loss,present_time))





if __name__=="__main__":
    train()
    pass
