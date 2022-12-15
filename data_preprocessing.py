import os
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Severson_Dataset_Training(Dataset):
    def __init__(self, train=True, pred_target='EOL', norm=True):
        """pred_target可更改為EOL或chargetime"""
        self.train = train
        self.pred_target = pred_target
        self.input, self.target = load_Severson(training=train, norm=norm)

    def __getitem__(self, index):
        target_id = 0 if self.pred_target=='EOL' else 1
        feature, target = self.input[index], self.target[index, target_id]
        return feature, target

    def __len__(self):
        return len(self.input)

    def visualize(self, index, feature_id):
        feature_list =  ['Voltage(V)', 'Discharge capacity(Ah)', 'Current', 'Temperature']
        curve = self.input[index, feature_id, :]
        plt.plot(np.arange(len(curve)), curve, c='red')
        plt.ylabel(feature_list[feature_id], fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.show()
        plt.close()


class Predictor1_Dataset(Dataset):
    def __init__(self, train=True, last_padding=True, fix_length=-1):
        self.train = train
        self.trn_input, self.trn_target = load_predictor1_features(training=True)
        self.val_input, self.val_target = load_predictor1_features(training=False)
        feature_scaler, target_scaler = StandardScaler(), StandardScaler()
        self.trn_input[:, :6, :] = feature_scaler.fit_transform(self.trn_input[:, :6, :].transpose((0, 2, 1)).reshape(-1, 6)).reshape(-1, 100, 6).transpose((0, 2, 1))
        self.val_input[:, :6, :] = feature_scaler.transform(self.val_input[:, :6, :].transpose((0, 2, 1)).reshape(-1, 6)).reshape(-1, 100, 6).transpose((0, 2, 1))
        self.trn_target, self.val_target = target_scaler.fit_transform(self.trn_target), target_scaler.transform(self.val_target)
        trn_size, val_size = len(self.trn_input), len(self.val_input)
        self.scaler_x, self.scaler_y = feature_scaler, target_scaler

        if last_padding: # full last padding
            aug_trn_input, aug_trn_target = [], []
            for i in range(trn_size):
                for cycle_length in range(100):
                    after_padding = self.trn_input[i].copy()
                    after_padding[:, cycle_length:] = after_padding[:, cycle_length].reshape(-1, 1).repeat(100-cycle_length, axis=1)
                    aug_trn_input.append(after_padding)
                    aug_trn_target.append(self.trn_target[i, :])
            self.trn_input, self.trn_target = np.stack(aug_trn_input, axis=0), np.stack(aug_trn_target, axis=0)
            
        if fix_length>-1:
            for i in range(trn_size):
                self.trn_input[i, :, fix_length:] = self.trn_input[i, :, fix_length].reshape(-1, 1).repeat(100-fix_length, axis=1)
            for i in range(val_size):
                self.val_input[i, :, fix_length:] = self.val_input[i, :, fix_length].reshape(-1, 1).repeat(100-fix_length, axis=1)

    def __getitem__(self, index):
        if self.train:
            feature, target = self.trn_input[index], self.trn_target[index]
            return feature, target
        feature, target = self.val_input[index], self.val_target[index]
        return feature, target

    def __len__(self):
        if self.train:
            return len(self.trn_input)
        return len(self.val_input)

    def get_scaler(self):
        return self.scaler_x, self.scaler_y


def load_Severson(training=True, norm=True):
    if training:
        feature, target = np.load('Severson_Dataset/Severson_TrnSet_input.npy'), np.load('Severson_Dataset/Severson_TrnSet_target.npy')
    else:
        feature, target = np.load('Severson_Dataset/Severson_ValSet_input.npy'), np.load('Severson_Dataset/Severson_ValSet_target.npy')
    if norm:
        return normalize(feature), normalize(target)
    else:
        return feature, target


def load_predictor1_features(training=False):
    folderpath = 'Severson_Dataset/discharge_model_each_cell/'
    filepath = sorted(os.listdir(folderpath))
    full_feature, full_target = [], []
    for i in range(len(filepath)//2):
        feature = np.load(folderpath+filepath[2*i])
        target = np.load(folderpath+filepath[2*i+1])
        full_feature.append(feature)
        full_target.append(target) 

    val_tar = np.load('Severson_Dataset/Severson_ValSet_target.npy')
    val_id = np.unique((val_tar[:, 0]+2)/val_tar[:, 1])
    # print(val_id.shape)
    trnset_x, valset_x = [], []
    trnset_y, valset_y = [], []
    for i, target in enumerate(full_target):
        if (target[0]/target[1]) in val_id:
            valset_x.append(full_feature[i])
            valset_y.append(full_target[i])
        else:
            trnset_x.append(full_feature[i])
            trnset_y.append(full_target[i])
    trnset_x, valset_x = np.stack(trnset_x, axis=0), np.stack(valset_x, axis=0)
    trnset_y, valset_y = np.stack(trnset_y, axis=0), np.stack(valset_y, axis=0)
    if training:
        # print(trnset_x.shape, trnset_y.shape)
        return trnset_x, trnset_y
    else:
        # print(valset_x.shape, valset_y.shape)
        return valset_x, valset_y
 

def get_scaler(pred_target='EOL'):
    """get the normalize scaler in training set"""
    assert pred_target=='EOL' or pred_target=='chargetime' or pred_target=='both' 
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    feature, target = np.load('Severson_Dataset/Severson_TrnSet_input.npy'), np.load('Severson_Dataset/Severson_TrnSet_target.npy')
    # scaler_x.fit(feature.reshape(-1, feature.shape[1]))
    scaler_x.fit(feature.transpose((0, 2, 1)).reshape(-1, 4))

    if pred_target=='EOL':
        target = target[:, 0].reshape(-1, 1)
    elif pred_target=='chargetime':
        target = target[:, 1].reshape(-1, 1)

    scaler_y.fit(target)
    return scaler_x, scaler_y


def normalize(data):
    scaler_x, scaler_y = get_scaler('both')
    c = data.shape[1] # channel
    if c>2: # feature
        data = scaler_x.transform(data.transpose((0, 2, 1)).reshape(-1, c)).reshape(-1, 500, c).transpose((0, 2, 1))
    else: # target
        data = scaler_y.transform(data.reshape(-1, c))
    return data
