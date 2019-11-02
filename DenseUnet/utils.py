import torch
import numpy as np
from torch.autograd import Variable,grad

def make_trainable(model,val):
    for p in model.parameters():
        p.requires_grad = val



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
