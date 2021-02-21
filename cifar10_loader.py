import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras.datasets import cifar10
import random
import matplotlib.pyplot as plt

class CIFAR10(Dataset):
    def __init__(self, size, train=True):
        self.size = size
        self.train = train
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()
        if train:
            self.X = train_X
            self.Y = train_y
        else:
            self.X = test_X
            self.Y = test_y
        # self.n = size//10
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
        img1 = self.X[j1]/255.0
        img2 = self.X[j2]/255.0
        img1 = np.transpose(img1, (2,0,1))
        img2 = np.transpose(img2, (2,0,1))
        return np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32), np.array([target], dtype=np.float32)

if __name__ == "__main__":
    train_set = CIFAR10(100, True)
    # print(train_set.list_idx[0])
    train_loader = DataLoader(train_set, 1, True)

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
