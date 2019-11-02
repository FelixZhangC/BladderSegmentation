# coding = utf-8
import numpy as np
import os
import imageio
import shutil

def GroundTruth2OneHot4Tumor(label):
    """
    just 255 = 1
    :param label: groundtruth (type: img)
    :return: onehot label (type: ndarry)
    """
    dim = label.shape[0]
    background = (label != 255).astype(int).reshape((dim, dim, 1))
    tumour = (label == 255).astype(int).reshape((dim, dim, 1))
    oneHotLabel = np.concatenate((background, tumour), 2)
    return oneHotLabel


def BatchGroundTruth2OneHot4Tumor(dirPath, savePath):
    for name in os.listdir(dirPath):
        label = np.array(imageio.imread(dirPath + "/" + name))
        if np.sum(label) > 0:
            oneHotLabel = GroundTruth2OneHot4Tumor(label)
            np.save(savePath + "/{0}.npy".format(name.split(".")[0]), oneHotLabel)

def GroundTruth2OneHot(label):
    """
    Convert label(0, 128, 255) with size W*H to onehot label (W*H*C)
    :param label: groundtruth (type: img)
    :return: onehot label (type: ndarry)
    """
    dim = label.shape[0]
    background = (label == 0).astype(int).reshape((dim, dim, 1))
    bladderWall = (label == 128).astype(int).reshape((dim, dim, 1))
    tumour = (label == 255).astype(int).reshape((dim, dim, 1))
    oneHotLabel = np.concatenate((background, bladderWall, tumour), 2)
    return oneHotLabel

def GroundTruth2OneHot2(label):
    """
    Convert label(0, 128, 255) with size W*H to onehot label (W*H*C)
    :param label: groundtruth (type: img)
    :return: onehot label (type: ndarry)
    """
    dim = label.shape[0]
    background = (label == 0).astype(int).reshape((dim, dim, 1))
    bladderWall = (label != 0).astype(int).reshape((dim, dim, 1))
    oneHotLabel = np.concatenate((background, bladderWall), 2)
    return oneHotLabel


def BatchGroundTruth2OneHot(dirPath, savePath):
    for name in os.listdir(dirPath):
        label = np.array(imageio.imread(dirPath + "/" + name))
        oneHotLabel = GroundTruth2OneHot(label)
        np.save(savePath + "/{0}.npy".format(name.split(".")[0]), oneHotLabel)


def BatchGroundTruth2OneHot2(dirPath, savePath):
    for name in os.listdir(dirPath):
        label = np.array(imageio.imread(dirPath + "/" + name))
        oneHotLabel = GroundTruth2OneHot2(label)
        np.save(savePath + "/{0}.npy".format(name.split(".")[0]), oneHotLabel)


def OneHot2GroundTruth(oneHotLabel):
    """
    Convert onehot label (W*H*C) to label(0, 128, 255) with size W*H
    :param oneHotLabel: onehot label (type: ndarry)
    :return: mask: groundtruth (type: img)
    """
    bladderWall = oneHotLabel[:, :, 1] * 128
    tumour = oneHotLabel[:, :, 2] * 255
    mask = (bladderWall + tumour).astype(np.uint8)
    return mask

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


def BatchOneHot2GroundTrutht(dirPath, savePath):
    for name in os.listdir(dirPath):
        label = np.load(dirPath + "/" + name)
        mask = OneHot2GroundTruth(label)
        imageio.imwrite(savePath + "/{0}.png".format(name.split(".")[0]), mask)


def GaussNoiseImg(img, mean=0.0, stddev=1.0):
    """
    Add Gasuss Noise to Image
    :param img:
    :param mean:
    :param stddev:
    :return:
    """
    noisyImg = img + np.random.normal(mean, stddev, img.shape)
    noisy_img_clipped = np.clip(noisyImg, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def BatchGaussNoiseImg(dirPath, savePath):
    for name in os.listdir(dirPath):
        img = np.array(imageio.imread(dirPath + "/" + name))
        noisyImg = GaussNoiseImg(img)
        imageio.imwrite(savePath + "/{0}_Gauss.png".format(name.split(".")[0]), noisyImg)


def AnisotropicFilter(image, t=10, k=15, lam=0.15):
    """
    Anisotropic Filter
    :param image:
    :param t:
    :param k:
    :param lam:
    :return:
    """
    img = image.copy()
    img = img.astype(float)
    imgFilter = np.zeros(img.shape, dtype=float)
    for i in range(t):
        for p in range(1, img.shape[0] - 1):
            for q in range(1, img.shape[1] - 1):
                NI = img[p - 1, q] - img[p, q]
                SI = img[p + 1, q] - img[p, q]
                EI = img[p, q - 1] - img[p, q]
                WI = img[p, q + 1] - img[p, q]
                cN = np.exp(- (NI * NI) / (k * k))
                cS = np.exp(- (SI * SI) / (k * k))
                cE = np.exp(- (EI * EI) / (k * k))
                cW = np.exp(- (WI * WI) / (k * k))
                imgFilter[p, q] = img[p, q] + lam * (cN * NI + cS * SI + cE * EI + cW * WI)
        img = imgFilter
    imgFilter = imgFilter.astype(np.uint8)
    return imgFilter


def BatchAnisotropicFilter(dirPath, savePath):
    for name in os.listdir(dirPath):
        img = np.array(imageio.imread(dirPath + "/" + name))
        anisotropicImg = AnisotropicFilter(img)
        imageio.imwrite(savePath + "/{0}_Anisotropic.png".format(name.split(".")[0]), anisotropicImg)


def Mixup(x1, x2, y1, y2, alpha=2):
    lam = np.random.beta(alpha, alpha)
    mixupX = lam * x1 + (1 - lam) * x2
    mixupY = lam * y1 + (1 - lam) * y2
    mixupX = mixupX.astype(np.uint8)
    mixupY = mixupY.astype(np.float32)
    return mixupX, mixupY


def Mixup0_5(x1, x2, y1, y2, alpha=2):
    lam = 0.5
    mixupX = lam * x1 + (1 - lam) * x2
    mixupY = lam * y1 + (1 - lam) * y2
    mixupX = mixupX.astype(np.uint8)
    mixupY = mixupY.astype(np.float32)
    return mixupX, mixupY


def BatchMixup(dirPath, labelPath, imgSavePath, labelSavePath):
    count = 1
    nameList = []
    for name in os.listdir(labelPath):
        nameList.append(name)
    for i in range(len(nameList)):
        for j in range(i+1, len(nameList)):
            x1 = imageio.imread(dirPath + "/{0}.png".format(nameList[i].split(".")[0]))
            x2 = imageio.imread(dirPath + "/{0}.png".format(nameList[j].split(".")[0]))
            y1 = np.load("D:/DATA/Bladder/CompareTest/OneHotLabelC2" + "/{0}.npy".format(nameList[i].split(".")[0]))
            y2 = np.load("D:/DATA/Bladder/CompareTest/OneHotLabelC2" + "/{0}.npy".format(nameList[j].split(".")[0]))
            mixupX, mixupY = Mixup(x1, x2, y1, y2, alpha=2)
            imageio.imwrite(imgSavePath + "/IM{0}_Mixup.png".format(str(count)), mixupX)
            np.save(labelSavePath + "/Label{0}_Mixup.npy".format(str(count)), mixupY)
            count += 1


def ImageGen4Tumor(img, label):
    """
    BackGround = 255
    :param img:
    :param label:
    :return:
    """
    background = (label != 0).astype(int)
    img = img.astype(int)
    elemWise = img * background
    elemWise = elemWise.astype(np.uint8)
    elemWise[elemWise == 0] = 255
    return elemWise

def BatchImageGen4Tumor(dirPath, labelPath, savePath):
    for name in os.listdir(dirPath):
        img = np.array(imageio.imread(dirPath + "/" + name))
        label = np.array(imageio.imread(labelPath + "/" + name))
        if np.sum(label) > 0:
            newImg = ImageGen4Tumor(img, label)
            imageio.imwrite(savePath + "/" + name, newImg)


def ImageGen4TumorBorder(img, label, padding=10):
    """
    BackGround = 0
    :param img:
    :param label:
    :return:
    """
    inds = np.where(label!=0)
    mask = np.zeros((512, 512), dtype=int)
    mask[inds[0].min()-padding:inds[0].max()+padding, inds[1].min()-padding:inds[1].max()+padding] = 1
    res = np.uint8(np.clip((1.68 * img + 10), 0, 255))
    elemWise = (res * mask).astype(np.uint8)
    return elemWise


def BatchImageGen4TumorBorder(dirPath, labelPath, savePath):
    for name in os.listdir(dirPath):
        img = np.array(imageio.imread(dirPath + "/" + name))
        label = np.array(imageio.imread(labelPath + "/" + name))
        if np.sum(label) > 0:
            newImg = ImageGen4TumorBorder(img, label, padding=10)
            imageio.imwrite(savePath + "/" + name, newImg)


def BatchNeighborMixup(dirPath, labelPath, imgSavePath, labelSavePath):
    count = 1
    idSet = set()
    nameDict = {}
    for name in os.listdir(dirPath):
        idSet.add(name.split("-")[0])
    for id in idSet:
        nameDict[id] = []
    for name in os.listdir(dirPath):
        nameDict[name.split("-")[0]].append(name)
    # print(nameDict)
    for key in nameDict.keys():
        picList = nameDict[key]
        for i in range(len(picList) - 1):
            x1 = imageio.imread(dirPath + "/" + picList[i])
            x2 = imageio.imread(dirPath + "/" + picList[i+1])
            y1 = np.load(labelPath + "/{0}.npy".format(picList[i].split(".")[0]))
            y2 = np.load(labelPath + "/{0}.npy".format(picList[i].split(".")[0]))
            for n in range(2):
                print("{0}-{1}".format(picList[i], picList[i+1]))
                mixupX, mixupY = Mixup(x1, x2, y1, y2, alpha=2)
                imageio.imwrite(imgSavePath + "/IM{0}_Mixup.png".format(str(count)), mixupX)
                np.save(labelSavePath + "/IM{0}_Mixup.npy".format(str(count)), mixupY)
                count += 1

def BatchMixup2(dirPath, labelPath, imgSavePath, labelSavePath):
    count = 1
    nameList = []
    for name in os.listdir(dirPath):
        nameList.append(name)
    for i in range(len(nameList)):
        for j in range(i+1, len(nameList)):
            x1 = imageio.imread(dirPath + "/" + nameList[i])
            x2 = imageio.imread(dirPath + "/" + nameList[j])
            y1 = np.load(labelPath + "/{0}.npy".format(nameList[i].split(".")[0]))
            y2 = np.load(labelPath + "/{0}.npy".format(nameList[j].split(".")[0]))
            mixupX, mixupY = Mixup(x1, x2, y1, y2, alpha=2)
            imageio.imwrite(imgSavePath + "/IM{0}_noBladder_Mixup.png".format(str(count)), mixupX)
            np.save(labelSavePath + "/IM{0}_noBladder_Mixup.npy".format(str(count)), mixupY)
            count += 1

