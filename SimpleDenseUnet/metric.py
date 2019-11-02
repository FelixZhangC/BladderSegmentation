import numpy as np


def OneHotMatric(inputs,target):
    inputs = inputs.cpu()
    target = target.cpu()
    inputs_arr = inputs.detach().numpy()
    target_arr = target.detach().numpy()
    inputs_arr = np.squeeze(inputs_arr)
    target_arr = np.squeeze(target_arr)
    #print("input shape:",inputs_arr.shape)
    #print("target shape:",target_arr.shape)

    inputs_arr = np.argmax(inputs_arr,axis=1)
    target_arr = np.argmax(target_arr,axis=1)

    TP = (inputs_arr==target_arr)& (inputs_arr!=0).astype(int)
    FN = (inputs_arr!=target_arr) & (inputs_arr==0).astype(int)
    FP = (inputs_arr!=target_arr) & (inputs_arr!=0).astype(int)
    TN = (inputs_arr==target_arr) & (inputs_arr==0).astype(int)
    return TP,FP,TN,FN



def PicMatric(inputs,target):
    inputs = inputs.cpu()
    target = target.cpu()
    inputs_arr = inputs.detach().numpy()
    target_arr = target.detach().numpy()
    inputs_arr = np.squeeze(inputs_arr)
    target_arr = np.squeeze(target_arr)
    #print("input shape:",inputs_arr.shape)
    #print("target shape:",target_arr.shape)

    inputs_arr=np.where(inputs_arr>=0.5,1,0)

    TP = (inputs_arr==target_arr)& (inputs_arr==1).astype(int)
    FN = (inputs_arr!=target_arr) & (target_arr==1).astype(int)
    FP = (inputs_arr!=target_arr) & (inputs_arr==1).astype(int)
    TN = (inputs_arr==target_arr) & (inputs_arr==0).astype(int)
    return TP,FP,TN,FN

def Dice(inputs,target):
    tp,fp,tn,fn = OneHotMatric(inputs,target)
    #print("tp sum:",np.sum(tp))
    #print("fp sum:",np.sum(fp))
    #print("fn sum:",np.sum(fn))
    dice = float(2.0*np.sum(tp)+1)/(2*np.sum(tp)+np.sum(fp)+np.sum(fn)+1)
    return dice


def PicDice(inputs_arr,target_arr):
    TP = (inputs_arr==target_arr) & (inputs_arr!=0).astype(int)
    FN = (inputs_arr!=target_arr) & (inputs_arr==0).astype(int)
    FP = (inputs_arr!=target_arr) & (inputs_arr!=0).astype(int)
    TN = (inputs_arr==target_arr) & (inputs_arr==0).astype(int)

    dice = float(2.0*np.sum(TP)+1)/(2*np.sum(TP)+np.sum(FP)+np.sum(FN)+1)
    return dice


def Jaccard(inputs,target):
    tp,fp,tn,fn = PicMatric(inputs,target)
    #print("tp sum:",np.sum(tp))
    #print("fp sum:",np.sum(fp))
    #print("fn sum:",np.sum(fn))
    jaccard = float(np.sum(tp)+1)/(np.sum(tp)+np.sum(fp)+np.sum(fn)+1)
    return jaccard

def Sensitivity(inputs,target):
    tp,fp,tn,fn = PicMatric(inputs,target)
    sensitivity = float(np.sum(tp)+1.0)/(np.sum(tp)+np.sum(fn)+1.0)
    return sensitivity

def Precision(inputs,target):
    tp,fp,tn,fn = PicMatric(inputs,target)
    precision = float(np.sum(tp)+1.0)/(np.sum(tp)+np.sum(fp)+1.0)
    return precision






