import logging
import os
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_logger

class AdultDataset(Dataset):
    """
    The UCI Adult dataset.
    """

    def __init__(self, root_dir, phase, tar_attr, priv_attr):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr
        self.npz_file = os.path.join(root_dir, 'adult_%s_%s.npz' % (tar_attr, priv_attr))
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1]
        self.adim = self.A.shape[1]

        if self.ydim != 1: 
            # one-hot encoding
            # change it to regression problem
            self.Y = np.argmax(self.Y, axis=1)
            self.Y = np.expand_dims(self.Y, -1)
            assert self.Y.shape[1] == 1 and len(self.Y.shape) == 2
            self.ydim = 1


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), \
                torch.from_numpy(self.Y[idx]).float(), \
                self.onehot_2_int(torch.from_numpy(self.A[idx]))

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2: # batch
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1: # one instance
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_YA_distribution(self):
        '''get the the mean Y when in different sensitive groups
        '''
        A_num_class = self.A.shape[1]
        assert A_num_class==2
        A_label= np.argmax(self.A, axis=1)
        A_0_mean_y = np.mean(self.Y[np.argwhere(A_label == 0).squeeze()])
        A_1_mean_y = np.mean(self.Y[np.argwhere(A_label == 1).squeeze()])
        return A_0_mean_y, A_1_mean_y

class CrimeDataset(Dataset):
    """
    The UCI commit dataset.
    """

    def __init__(self, root_dir, phase):
        self.npz_file = os.path.join(root_dir, 'communities_crime.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.Y = np.expand_dims(self.Y, axis=-1)

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1]
        self.adim = 2


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), \
                torch.from_numpy(self.Y[idx]).float(), \
                self.A[idx]

    def get_YA_distribution(self):
        '''get the the mean Y when in different sensitive groups
        '''
        A_num_class = self.adim
        assert A_num_class==2
        A_label= self.A
        A_0_mean_y = np.mean(self.Y[np.argwhere(A_label == 0).squeeze()])
        A_1_mean_y = np.mean(self.Y[np.argwhere(A_label == 1).squeeze()])
        return A_0_mean_y, A_1_mean_y

class COMPAS(Dataset):
    """
    The COMPAS dataset.
    """
    def __init__(self, data):
        """
        :param data:    Numpy array that contains all the data.
        """
        # Data, label and sensitive attribute partition.
        self.insts = data[:, 1:].astype(np.float32)
        self.labels = data[:, 0].astype(np.int64)
        self.attrs = data[:, 5].astype(np.int64)
        self.xdim = self.insts.shape[1]

        self.labels = np.expand_dims(self.labels, axis=-1)

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]).float(), \
               torch.tensor(self.labels[idx]).float(), \
               torch.tensor(self.attrs[idx])

class LawSchoolDataset(Dataset):
    """
    The UCI law dataset.
    """

    def __init__(self, root_dir, phase):
        self.npz_file = os.path.join(root_dir, 'law_school.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.Y = np.expand_dims(self.Y, axis=-1)

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1]
        self.adim = 2


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), \
                torch.from_numpy(self.Y[idx]).float(), \
                self.A[idx]

    def get_YA_distribution(self):
        '''get the the mean Y when in different sensitive groups
        '''
        A_num_class = self.adim
        assert A_num_class==2
        A_label= self.A
        A_0_mean_y = np.mean(self.Y[np.argwhere(A_label == 0).squeeze()])
        A_1_mean_y = np.mean(self.Y[np.argwhere(A_label == 1).squeeze()])
        return A_0_mean_y, A_1_mean_y

class InsuranceDataset(Dataset):
    """
    The medical cost prediction dataset.
    """

    def __init__(self, root_dir, phase):
        self.npz_file = os.path.join(root_dir, 'insurance.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.Y = np.expand_dims(self.Y, axis=-1)

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1]
        self.adim = 2


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), \
                torch.from_numpy(self.Y[idx]).float(), \
                self.A[idx]

    def get_YA_distribution(self):
        '''get the the mean Y when in different sensitive groups
        '''
        A_num_class = self.adim
        assert A_num_class==2
        A_label= self.A
        A_0_mean_y = np.mean(self.Y[np.argwhere(A_label == 0).squeeze()])
        A_1_mean_y = np.mean(self.Y[np.argwhere(A_label == 1).squeeze()])
        return A_0_mean_y, A_1_mean_y

if __name__ == '__main__':
    logger = get_logger("dataset")
    ######## test pytorch adult dataset ######### 
    root_dir = 'data'
    tar_attr, priv_attr='income', 'sex' # income education-num
    npz_file = os.path.join(root_dir, 'adult_%s_%s.npz' % (tar_attr, priv_attr))
    data = np.load(npz_file)
    
    logger.info("Adult dataset")
    logger.info("tar_attr: %s, priv_att: %s"% (tar_attr, priv_attr))
    X = data["x_train"]
    Y = data["y_train"]
    A = data["attr_train"]; A = np.argmax(A, axis=1)
    logger.info('tar_attr, priv_attr=%s, %s'%(tar_attr, priv_attr))
    logger.info("In training set:")
    logger.info("max min for X are: %.6f, %.6f"%(np.amin(X), np.amax(X)))
    logger.info("max min for Y are: %.6f, %.6f"%(np.amin(Y), np.amax(Y)))
    logger.info("max min for A are: %.6f, %.6f"%(np.amin(A), np.amax(A)))
    logger.info("0/1 for A are: %d, %d"%(len(A)-np.sum(A), np.sum(A)))
    # from collections import Counter
    # print(Counter(Y.squeeze().astype(str).tolist()))

    trainset = AdultDataset(root_dir=root_dir, phase='train', tar_attr=tar_attr, priv_attr=priv_attr)
    testset = AdultDataset(root_dir=root_dir, phase='test', tar_attr=tar_attr, priv_attr=priv_attr)
    print(len(trainset), trainset[:100][0].shape, trainset[:100][1].shape, trainset[:100][2].shape)
    print(len(testset), testset[:100][0].shape, testset[:100][1].shape, testset[:100][2].shape)
    print(trainset[:100][0].dtype, trainset[:100][1].dtype, trainset[:100][2].dtype)

    A_0_mean_y, A_1_mean_y = trainset.get_YA_distribution()
    logger.info("In training set, the mean value of Y in different group:(%.6f,%.6f)"%(A_0_mean_y,A_1_mean_y))

    A_0_mean_y, A_1_mean_y = testset.get_YA_distribution()
    logger.info("In test set, the mean value of Y in different group:(%.6f,%.6f)"%(A_0_mean_y,A_1_mean_y))
    #######################################################

    ################ test pytorch crime dataset #################
    print('\n\n')
    logger.info("Crime dataset")
    root_dir = 'data'
    trainset = CrimeDataset(root_dir=root_dir, phase='train')
    testset = CrimeDataset(root_dir=root_dir, phase='test')
    X = trainset.X
    Y = trainset.Y
    A = trainset.A
    logger.info("In training set:")
    logger.info("max min for X are: %.6f, %.6f"%(np.amin(X), np.amax(X)))
    logger.info("max min for Y are: %.6f, %.6f"%(np.amin(Y), np.amax(Y)))
    logger.info("max min for A are: %.6f, %.6f"%(np.amin(A), np.amax(A)))
    print("len(trainset), len(testset): ",len(trainset), len(testset))
    print(len(trainset), trainset[:100][0].shape, trainset[:100][1].shape, trainset[:100][2].shape)
    print(len(testset), testset[:100][0].shape, testset[:100][1].shape, testset[:100][2].shape)
    print(trainset[:100][0].dtype, trainset[:100][1].dtype, trainset[:100][2].dtype)

    A_0_mean_y, A_1_mean_y = trainset.get_YA_distribution()
    logger.info("In training set, the mean value of Y in different group:(%.6f,%.6f)"%(A_0_mean_y,A_1_mean_y))

    A_0_mean_y, A_1_mean_y = testset.get_YA_distribution()
    logger.info("In test set, the mean value of Y in different group:(%.6f,%.6f)"%(A_0_mean_y,A_1_mean_y))
    #######################################################

    ################ test pytorch compas dataset #################
    compas = pd.read_csv("data/propublica.csv").values
    labels = compas[:, 0].astype(np.int64)
    attrs = compas[:, 5].astype(np.int64)
    print(labels.shape, attrs.shape)
    print(labels[attrs==0].shape, np.sum(labels[attrs==0]))
    print(labels[attrs==1].shape, np.sum(labels[attrs==1]))