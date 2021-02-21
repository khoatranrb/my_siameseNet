import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os

class Omniglot(Dataset):
    def __init__(self, name, train):
        PATH = 'Omniglot Dataset/images_evaluation'
        if train:
            PATH = 'Omniglot Dataset/images_background'
        self.path = PATH
        self.train = train
        self.name = name
        self.list_classes = os.listdir(os.path.join(PATH, name))
        self.list_images = []
        for _cl in self.list_classes:
            self.list_images.append(os.listdir(os.path.join(PATH, name, _cl)))
    def __len__(self):
        return 19*len(self.list_classes)
    def __getitem__(self, idx):
        target = random.randint(0,1)
        idx_class1 = idx//len(self.list_classes)
        idx_class2 = idx_class1
        if not target:
            while idx_class2 == idx_class1:
                idx_class2 = random.randint(0, len(self.list_classes)-1)
        i1 = idx%20
        step = 1
        if self.train:
            step = random.randint(0,19)
        i2 = (i1+step)%20

        path1 = os.path.join(self.path, self.name, self.list_classes[idx_class1], self.list_images[idx_class1][i1])
        path2 = os.path.join(self.path, self.name, self.list_classes[idx_class2], self.list_images[idx_class2][i2])

        img1 = cv2.imread(path1)
        img1 = cv2.resize(img1, (32,32))
        img1 = np.transpose(img1, (2,0,1))/255.0
        img2 = cv2.imread(path2)
        img2 = cv2.resize(img2, (32,32))
        img2 = np.transpose(img2, (2,0,1))/255.0

        return np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32), np.array([target], dtype=np.float32)

if __name__=="main":
    trainset = Data('Latin', True)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)

    train_iter = iter(train_loader)

    x, y, z = next(train_iter)

    print(x.shape, y.shape, z)
    x = x[0].numpy()
    y = y[0].numpy()
    plt.imshow(x[:,:,[2,1,0]])
    plt.show()
    plt.imshow(y[:,:,[2,1,0]])
    plt.show()
