import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def BCE_loss(inputs,targets):
    loss = F.binary_cross_entropy(inputs,targets)
    return loss
def Cross_Entropy2d(inputs,targets):
    # 网络最后一层已经写了softmax，此时得到的inputs就已经是概率形式得了
    #print("inputs shape:",inputs.shape)
    #print("targets shape:",targets.shape)
    inputs = torch.clamp(inputs,min=1e-7,max=1.0-(1e-7))
    celoss =-(torch.log(inputs)*targets)
    #print("celoss shape:",celoss.shape)
    celoss = torch.sum(celoss,dim=(1,2,3))
    #print("celoss shape:",celoss.shape)
    #print(celoss)
    celoss = torch.mean(celoss)
    #print(celoss)
    celoss = celoss/(512.0*512.0)
    #print(celoss)
    return celoss


class myWeightedDiceLoss4Organs(nn.Module):
    def __init__(self, organIDs = [0,1,2], organWeights=[340,3,1]):
        super(myWeightedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
#         pass

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1
        """
        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

        inputSZ = inputs.size() #it should be sth like NxCxHxW

        inputs = F.softmax(inputs, dim=1)
#         _, results_ = inputs.max(1)
#         results = torch.squeeze(results_) #NxHxW

        numOfCategories = inputSZ[1]
        assert numOfCategories==len(self.organWeights), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs


        # target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
        # targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
        # targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
        targets_one_hot = Variable(targets)



        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

        for organID in range(0, numOfCategories):
#             target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
#             result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
            target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
            result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)

    #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            intersect = torch.sum(intersect_vec)
    #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
    #         print type(result_sum)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + (two*eps)

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            IoU = intersect / union
    #             out = torch.add(out, IoU.data*2)
            out = out + self.organWeights[organID] * (one - two*IoU)
    #         print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    # organID, union.data[0], intersect.data[0], target_sum.data[0], result_sum.data[0], IoU.data[0])

        denominator = Variable(torch.cuda.FloatTensor(1).fill_(sum(self.organWeights)))
#         denominator = Variable(torch.cuda.FloatTensor(1).fill_(numOfCategories))

        out = out / denominator
    #     print type(out)
        return out


def WeightedDiceLoss(inputs,targets):
    organIDs = [0,1,2]
    organWeights=[340,3,1]
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

    inputsize = inputs.size()
    numOfCategories = inputsize[1]


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



