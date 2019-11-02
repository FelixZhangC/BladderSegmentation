import torch
import numpy as np
from torch.autograd import Variable,grad

def make_trainable(model,val):
    for p in model.parameters():
        p.requires_grad = val




def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):
    BATCH=real_data.size()[0]
    alpha = torch.rand(BATCH, 1)
    #print(alpha.size(),real_data.size())
    alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs

class ToFloatTensor(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs)).float()
