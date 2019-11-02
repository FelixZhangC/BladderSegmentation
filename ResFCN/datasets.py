import torch
import torchvision
from torch.utils import data
import os
import imageio
import numpy as np

class BladderOneHot(data.Dataset):
    def __init__(self,imageroot,labelroot,img_transform=None,label_transform=None,img_label_transform=None):
        super(BladderOneHot,self).__init__()
        self.files = []
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_label_transform = img_label_transform
        print("length of Dataset:",len(os.listdir(imageroot)))
        for name in os.listdir(imageroot):
            name = os.path.splitext(name)[0]
            #name = name[2:]
            img_name = os.path.join(imageroot,"%s.png"%name)
            label_name = os.path.join(labelroot,"%s.npy"%name)

            self.files.append({
                "img":img_name,
                "label":label_name
            })

    def __getitem__(self, item):
        datafile = self.files[item]
        img_name = datafile["img"]
        label_name = datafile["label"]

        img_arr = imageio.imread(img_name)
        img_arr = np.array(np.expand_dims(img_arr,axis=0))
        label_arr = np.load(label_name)
        label_arr = label_arr.transpose(2,0,1)
        #print("label_arr shape:",label_arr.shape)
        #print("img_arr shape:",img_arr.shape)
        
        if self.img_label_transform is not None:
            img_arr,label_arr = self.img_label_transform(img_arr,label_arr)

        if self.label_transform is not None:
            label_arr = self.label_transform(label_arr)

        if self.img_transform is not None:
            img_arr = self.img_transform(img_arr)

        #print("type img:",type(img_arr))
        #print("type label:",type(label_arr))
        return img_arr,label_arr

    def __len__(self):
        return len(self.files)






if __name__ == "__main__":
    pass
