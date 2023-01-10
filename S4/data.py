import numpy as np
import torch


def mnist():
    # exchange with the corrupted mnist dataset
    train = np.load('corruptmnist/train_0.npz',allow_pickle=True) #train['images'] (5000,28,28), train['labels']
    test = np.load('corruptmnist/test.npz',allow_pickle=True) #test['images'], test['labels']

    train2 = list(zip(train['images'].reshape(-1, 1, 28, 28).astype(np.float32), train['labels']))
    test2 = list(zip(test['images'].reshape(-1, 1, 28, 28).astype(np.float32), test['labels']))

    return train2, test2

train,test=mnist()



