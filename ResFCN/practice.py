import numpy as np
import os
import imageio
import shutil

def OneHot2GroundTruth2(oneHotLabel):
    """
    Convert onehot label (W*H*C) to label(0, 128, 255) with size W*H
    :param oneHotLabel: onehot label (type: ndarry)
    :return: mask: groundtruth (type: img)
    """
    # bladderWall = oneHotLabel[:, :, 0] * 128
    tumour = oneHotLabel[:, :, 1] * 255
    mask = tumour.astype(np.uint8)
    return mask


def FindImages(testpath,totalsetpath,totalmaskpath,dst1,dst2):
    testlist = os.listdir(testpath)
    for testpic in testlist:
        testarr = imageio.imread(os.path.join(testpath,testpic))
        for pic in os.listdir(totalsetpath):
            picarr = imageio.imread(os.path.join(totalsetpath,pic))
            pixelsum = np.sum(testarr == picarr)
            if pixelsum == 512*512:
                print(testpic)
                shutil.copyfile(os.path.join(totalsetpath,pic),os.path.join(dst1,testpic))
                shutil.copyfile(os.path.join(totalmaskpath,pic),os.path.join(dst2,testpic))
                break



if __name__ == "__main__":
    testpath = "./GameTest/Image/"
    totalsetpath = "../GithubData/Image/"
    totalmaskpath = "../GithubData/Label/"
    dst1 = "../0816/TestSetWithTrueMask/Images/"
    dst2 = "../0816/TestSetWithTrueMask/Labels/"
    if not os.path.exists(dst1):
        os.makedirs(dst1)
    if not os.path.exists(dst2):
        os.makedirs(dst2)
    FindImages(testpath,totalsetpath,totalmaskpath,dst1,dst2)


    pass



