import numpy as np
import skimage
import imageio
import os

from skimage import measure
from skimage import io


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




if __name__ == "__main__":

    root = "./0819Pred/G_epoch300/"
    saveroot = "./0819Pred/G_epoch300process/"
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    for pic in os.listdir(root):
        print(pic)
        picarr = imageio.imread(root+pic)
        outp = FindRegion(picarr,1000)
        outp = np.array(outp,dtype='uint8')
        #print(np.unique(outp))
        imageio.imwrite(saveroot+pic,outp)


    pass
