import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras.datasets import mnist
import random
import matplotlib.pyplot as plt
import cv2

class MNIST(Dataset):
    def __init__(self, size, train=True):
        self.size = size
        self.train = train
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        if train:
            self.X = train_X
            self.Y = train_y
        else:
            self.X = test_X
            self.Y = test_y
        self.list_idx = []
        for i in range(10):
            self.list_idx.append(np.where(self.Y==i)[0])
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        i1 = random.randint(0,9)
        max_idx1 = self.list_idx[i1].shape[0]
        target = random.randint(0,1)
        i2 = i1
        if target==0:
            while i2==i1:
                i2 = random.randint(0,9)
        max_idx2 = self.list_idx[i2].shape[0]
        j1 = random.randint(0,max_idx1)
        j1 = self.list_idx[i1][j1]
        j2 = random.randint(0,max_idx2)
        j2 = self.list_idx[i2][j2]
        img1 = cv2.resize(self.X[j1], (32,32))/255.0
        img2 = cv2.resize(self.X[j2], (32,32))/255.0
        img1 = img1[np.newaxis, ...]
        img1 = np.concatenate([img1]*3, axis=0)
        img2 = img2[np.newaxis, ...]
        img2 = np.concatenate([img2]*3, axis=0)
        return np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32), np.array([target], dtype=np.float32)

if __name__ == "__main__":
    train_set = MNIST(100, True)
    train_loader = DataLoader(train_set, 2, True)

    train_iter = iter(train_loader)
    a, b, c = next(train_iter)
    print(c)
    print(a.shape, a.dtype)
    img = a[0].numpy()
    img = np.transpose(img, (1,2,0))
    plt.imshow(img[:,:,[2,1,0]])
    plt.show()
    img = b[0].numpy()
    img = np.transpose(img, (1,2,0))
    plt.imshow(img[:,:,[2,1,0]])
    plt.show()
