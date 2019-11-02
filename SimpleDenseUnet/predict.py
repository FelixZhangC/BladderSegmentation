import torch
import os
import imageio
import numpy as np
import shutil
from skimage import measure
from collections import Counter

from Nets.DenseUnet import DenseNet2d
from Nets.NewDenseNet import DenseUNet
from metric import PicDice

def FindRegion(inputs,threld):
    inputpic = inputs.copy()
    inputpic[inputpic==128] = 1
    inputpic[inputpic==255] = 1

    labelregion = measure.label(inputpic,connectivity=2)
    alllabellist = np.unique(labelregion)
    #print(alllabellist)
    regionlist = measure.regionprops(labelregion)
    #print(type(regionlist))

    maxregion = 0
    maxlabel = 0
    labellist = []

    for i in range(len(regionlist)):
        if regionlist[i].area > threld:
            labellist.append(regionlist[i].label)

    for label in alllabellist:
        if label not in labellist:
            labelregion[labelregion==label] = 0

    labeloutput = np.where(labelregion!=0,1,0)

    outputarr = labeloutput*inputs
    return outputarr



    labelarr = np.where(labelregion==maxlabel,1,0)

    return labelarr

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

    #realmaskpath = "../TestSetWithTrueMask/Labels/"

    # quesimg = "../TestSetN/QuesDiceImgs/"
    # if quesimg != None and not os.path.exists(quesimg):
    #     os.makedirs(quesimg)
    # corrimg = "../TestSet/CorrDiceImgs/"
    # if corrimg != None and not os.path.exists(corrimg):
    #     os.makedirs(corrimg)

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

        #predarr[predarr==1] = 128
        predarr[predarr==1] = 128
        predarr[predarr==2] = 255
        predarr = np.array(predarr,dtype='uint8')
        imageio.imwrite(os.path.join(masksavepath,pic),predarr)

def PredAssample(model1,model2,predpicpath,masksavepath,threld,device =None):
    if not os.path.exists(masksavepath):
        os.makedirs(masksavepath)
    piclist = os.listdir(predpicpath)
    for pic in piclist:
        print(pic)
        picname = pic.split('.')[0]
        picpath = os.path.join(predpicpath,pic)
        picarr = imageio.imread(picpath)
        picarr = np.expand_dims(picarr,axis=0)
        picarr = np.expand_dims(picarr,axis=0)

        pictensor = torch.Tensor(picarr)
        pictensor = pictensor.to(device)

        predpic1 = model1(pictensor)
        predpic2 = model2(pictensor)

        predpic1 = torch.squeeze(predpic1)
        predarr1 = predpic1.cpu().detach().numpy()
        predarr1 = np.argmax(predarr1,axis=0)
        predpic2 = torch.squeeze(predpic2)
        predarr2 = predpic2.cpu().detach().numpy()
        predarr2 = np.argmax(predarr2,axis=0)

        predarr1[predarr1==1] = 128
        predarr1[predarr1==2] = 255
        #predarr1 = np.array(predarr1,dtype='uint8')
        predarr2[predarr2==1] = 128
        predarr2[predarr2==2] = 255
        #predarr2 = np.array(predarr2,dtype='uint8')

        # Arr1 = FindRegion(predarr1,1000)
        # Arr2 = FindRegion(predarr2,1000)
        Arr1 = predarr1
        Arr2 = predarr2



        #print("Arrq", Arr1.max())
        outputarr = np.zeros(np.shape(Arr1))
        outputarr[(Arr1==128) | (Arr2==128)] = 128
        #print(np.where(outputarr[(Arr1==255) | (Arr2==1)]))

        outputarr[(Arr1==255) | (Arr2==255)] = 255

        #outputarr[outputarr==1] =128
        #outputarr[outputarr==2] = 255

        #print(np.unique(outputarr))

        outputarr = np.array(outputarr,dtype='uint8')

        outputarr = FindRegion(outputarr,threld)
        outputarr = np.array(outputarr,dtype='uint8')

        imageio.imwrite(os.path.join(masksavepath,pic),outputarr)

def PredVote(model1,model2,model3,predpicpath,masksavepath,threld,device=None):
    if not os.path.exists(masksavepath):
        os.makedirs(masksavepath)
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

        predpic1 = model1(pictensor)
        predpic2 = model2(pictensor)
        predpic3 = model3(pictensor)

        predpic1 = torch.squeeze(predpic1)
        predarr1 = predpic1.cpu().detach().numpy()
        predarr1 = np.argmax(predarr1,axis=0)
        predpic2 = torch.squeeze(predpic2)
        predarr2 = predpic2.cpu().detach().numpy()
        predarr2 = np.argmax(predarr2,axis=0)
        predpic3 = torch.squeeze(predpic3)
        predarr3 = predpic3.cpu().detach().numpy()
        predarr3 = np.argmax(predarr3,axis=0)


        predThreeArr = np.stack([predarr1,predarr2,predarr3],axis=0)

        predArr = np.zeros((512,512))
        for i in range(512):
            for j in range(512):
                predlist = predThreeArr[:,i,j].tolist()
                #print(predlist)
                if predlist[0]!= predlist[1] and predlist[0]!=predlist[2] and predlist[1]!=predlist[2]:
                    predArr[i,j]=2
                realpred = Counter(predlist).most_common(1)[0][0]
                predArr[i,j] = realpred

        predArr[predArr==1] = 128
        predArr[predArr==2] = 255
        outputarr = np.array(predArr,dtype='uint8')

        outputarr = FindRegion(outputarr,threld)
        outputarr = np.array(outputarr,dtype='uint8')

        imageio.imwrite(os.path.join(masksavepath,pic),outputarr)



# if __name__ == "__main__":
#     modelpath = "./checkpoint/"
#
#     for model in os.listdir(modelpath):
#         modelroot = modelpath + model
#         modelname = model.split('.')[0]
#         predpath = "../TestSetNoMask/Image/"
#         maskpath = os.path.join("../TestSetNoMask/KingDataDenseDoubleLoss/",modelname)
#
#         device = "cuda"
#         model = DenseNet2d().to(device)
#         print(modelroot)
#         model.load_state_dict(torch.load(modelroot))
#         PredictOneHot(model,predpath,maskpath,device)





if __name__ == "__main__":
    model1path = "./checkpoint/G_epoch200.pth"
    #model2path = "./KingDataNewDenseModel/G_epoch20.pth"
    model2path = "./checkpoint2/G_epoch200.pth"

    predpath = "../FinalTestData/"
    maskpath = "../0823FinalPredDense1000/"

    device = "cuda"
    model1 = DenseNet2d(num_classes=3).to(device)
    model1.load_state_dict(torch.load(model1path))

    model2 = DenseNet2d(num_classes=3).to(device)
    model2.load_state_dict(torch.load(model2path))

    PredAssample(model1,model2,predpicpath=predpath,masksavepath=maskpath,threld=1000,device=device)




