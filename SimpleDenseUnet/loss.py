import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def WeightedDiceLoss(inputs,targets):
    organIDs = [0,1,2]
    organWeights=[1,3,340]
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

    inputsize = inputs.size()
    numOfCategories = inputsize[1]
    inputs = F.softmax(inputs,dim=1)


    inputs_one_hot = inputs
    targets_one_hot = targets

    out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)

    for organID in range(0,numOfCategories):
        target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
        input = inputs_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1) #for 2D or 3D

        intersect_vec = input * target
        intersect = torch.sum(intersect_vec)

        result_sum = torch.sum(input)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (two*eps)

        IoU = intersect / union

        out = out + organWeights[organID] * (one-two*IoU)

    denominator = Variable(torch.cuda.FloatTensor(1).fill_(sum(organWeights)))

    out = out / denominator
    return out


def WeightedCrossEntropy(inputs,targets):
    pixelweight = torch.Tensor(np.array([1,3,340]))
    pixelweight = pixelweight.cuda()
    n,c,h,w = inputs.size()

    targets = torch.argmax(targets,dim=1)
    targets = torch.unsqueeze(targets,dim=1)
    targets = targets.transpose(1,2).transpose(2,3).contiguous()
    targets = targets.view(-1)
    #print(targets.shape)

    predict = inputs.transpose(1,2).transpose(2,3).contiguous()
    predict = predict.view(-1,c)
    #print(predict.shape)

    loss = F.cross_entropy(predict,targets,weight=pixelweight)
    return loss





if __name__ == "__main__":
    import numpy as np
    a1 = [[0.3,0.1,0.9],
             [0.5,0.2,0.1],
             [0.1,0.4,0.2]]
    b1 = [[0.7,0.9,0.1],
         [0.5,0.8,0.9],
         [0.9,0.6,0.8]]

    a2 = [[0.7,0.1,0.9],
             [0.9,0.2,0.1],
             [0.1,0.4,0.2]]
    b2 = [[0.3,0.9,0.1],
         [0.1,0.8,0.9],
         [0.9,0.6,0.8]]
    input1=[]
    input1.append(a1)
    input1.append(b1)
    input2=[]
    input2.append(a2)
    input2.append(b2)
    input =[]
    input.append(input1)
    input.append(input2)
    input = np.array(input)

    aa1 = [[1,0,0],
          [0,1,0],
          [1,0,0]]
    bb1 = [[0,1,1],
          [1,0,1],
          [0,1,1]]
    aa2 = [[0,0,0],
          [0,1,0],
          [1,0,0]]
    bb2 = [[1,1,1],
          [1,0,1],
          [0,1,1]]
    target1 = []
    target1.append(aa1)
    target1.append(bb1)
    target2 = []
    target2.append(aa2)
    target2.append(bb2)
    target=[]
    target.append(target1)
    target.append(target2)
    target = np.array(target)

    input = torch.Tensor(input)
    target = torch.Tensor(target)
    print("input shape:",input.shape)

    loss = Cross_Entropy2d(input,target)
    print("loss : %.10f ; "%(loss.detach().numpy()))
    print(np.log(0.3))
