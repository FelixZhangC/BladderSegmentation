import numpy as np
import os
import imageio
import shutil



def CalcuPixelRate(root):
    piclist = os.listdir(root)
    print(len(piclist))
    class0 = 0
    class1 = 0
    class2 = 0
    for pic in piclist:
        picarr = imageio.imread(root+pic)
        class0 += np.sum(picarr==0)
        class1 += np.sum(picarr==128)
        class2 += np.sum(picarr==255)
    print(class0)
    print(class1)
    print(class2)
    return class0,class1,class2

def TransImages(root1,root2,dst):
    totallist = os.listdir(root1)
    pictruemasklist = os.listdir(root2)
    for pic in totallist:
        if pic not in pictruemasklist:
            shutil.copy(os.path.join(root1,pic),os.path.join(dst,pic))

def FindImages(testpath,testmaskpath,totalsetpath,dst1,dst2,dst3,dst4):
    testlist = os.listdir(testpath)
    for testpic in testlist:
        print(testpic)
        testarr = imageio.imread(os.path.join(testpath,testpic))
        for pic in os.listdir(totalsetpath):
            picarr = imageio.imread(os.path.join(totalsetpath,pic))
            pixelsum = np.sum(testarr == picarr)
            if pixelsum == 512*512:
                shutil.copyfile(os.path.join(testpath,testpic),os.path.join(dst1,testpic))
                shutil.copyfile(os.path.join(testmaskpath,testpic),os.path.join(dst2,testpic))
                break
            # else:
            #     shutil.copyfile(os.path.join(testpath,testpic),os.path.join(dst3,testpic))
            #     shutil.copyfile(os.path.join(testmaskpath,testpic),os.path.join(dst4,testpic))


def FindSimilarImg(testroot,root):
    for testpic in os.listdir(testroot):
        testpicarr = imageio.imread(testroot+testpic)
        similarDict = {}
        for pic in os.listdir(root):
            picarr = imageio.imread(root+pic)
            similarate = np.sum(testpicarr==picarr)
            similarDict[pic] = similarate

        DictSortedList = sorted(similarDict.items(),key = lambda x:x[1],reverse=True)
        print(testpic,DictSortedList[0:30])




if __name__ == "__main__":
    testpath = "../TestSetNoMask/Image/"
    totalpath = "../TestSetNoMask/Image/"
    FindSimilarImg(testpath,totalpath)


    pass



