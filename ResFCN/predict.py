import torch
import os
import imageio
import numpy as np
import shutil

from Nets.ResFCNnet import ResSegNet
from metric import PicDice

def PredictOneHot(model,predpicpath,masksavepath=None,device=None):
    '''
    :param model: 模型
    :param predpicpath: 预测图片地址
    :param predpicmaskpath: 预测图片中原来有mask的
    :param masksavepath: 预测得到的mask地址
    :param outdatapicpath: 预测数据中外部数据的备份
    :param outdatamaskpath: 预测数据中外部数据的mask
    :return:
    '''
    if not os.path.exists(masksavepath):
        os.makedirs(masksavepath)

    realmaskpath = "../TestSetWithTrueMask/Labels/"
    piclist = os.listdir(predpicpath)
    for pic in piclist:
        #print(pic)
        picname = pic.split('.')[0]
        picpath = os.path.join(predpicpath,pic)
        picarr = imageio.imread(picpath)
        picarr = np.expand_dims(picarr,axis=0)
        picarr = np.expand_dims(picarr,axis=0)

        pictensor = torch.Tensor(picarr)
        pictensor = pictensor.to(device)
        predpic = model(pictensor)

        predpic = torch.squeeze(predpic)
        predarr = predpic.cpu().detach().numpy()



        predarr = np.argmax(predarr,axis=0)


        realmask = imageio.imread(realmaskpath+pic)
        realmask[realmask==255] = 2
        realmask[realmask==128] = 1
        dice = PicDice(predarr,realmask)
        print(pic,dice)
        #predarr[predarr==1] = 128
        predarr[predarr==1] = 128
        predarr[predarr==2] = 255
        predarr = np.array(predarr,dtype='uint8')
        imageio.imwrite(os.path.join(masksavepath,pic),predarr)


if __name__ == "__main__":
    modelpath = "./G_epoch_100.pth"
    predpath = "../TestSetWithTrueMask/Images/"

    maskpath = "../TestSetWithTrueMask/ResFcnThreePred/"

    device = "cuda"
    model = ResSegNet(in_channels=1,out_channels=3).to(device)
    model.load_state_dict(torch.load(modelpath))
    PredictOneHot(model,predpath,maskpath,device)
    #PredictOneHot(model,predpicpath=predpath,masksavepath=maskpath,device=device)
