import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class ESASegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", target_channels = None, train_length = '3_months'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # training and validation data
        df = pd.read_csv(data_path + '/' + train_length +'.train.csv')

        # Identify telemetry channels and anomaly labels
        self.all_channels = [col for col in df.columns if col.startswith('channel_')]
        self.telecommand_cols = [col for col in df.columns if col.startswith('telecommand_')]

        # Select target channels
        if target_channels is None:
            self.target_channels = self.all_channels
        else:
            self.target_channels = [ch for ch in target_channels if ch in self.all_channels]

        print(f"Total channels: {len(self.all_channels)}")
        print(f"Using channels: {len(self.target_channels)}")

        # Extract anomaly labels (per channel)
        self.label_columns = [f'is_anomaly_{ch}' for ch in self.target_channels]
        self.labels = df[self.label_columns].values.astype(np.float32)

        # Extract telemetry data
        data = df[self.target_channels].values.astype(np.float32)

        # normalizing data
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # splitting train into train and validation
        data_len = len(data)

        self.train = data[:int(data_len * 0.8)]

        self.val = data[int(data_len * 0.8):]
        self.val_labels = self.labels[int(data_len * 0.8):]

        # test data
        test_df = pd.read_csv(data_path + '/' + '84_months.test.csv')

        test_data = test_df[self.target_channels].values.astype(np.float32)
        self.test_labels = test_df[self.label_columns].values.astype(np.float32)

        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        print("test:", self.test.shape)
        print("val:", self.val.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
            #   np.float32(self.val_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def get_loader_segment(data_path, batch_size, win_size= 100, step = 100, mode = 'train', dataset = '3_months'):

    dataset = ESASegLoader(data_path, win_size, step, mode=mode, train_length=dataset)

    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
    

if __name__ == "__main__":
    data_path = '../../ESA-ADB-mdm/data/preprocessed_subset/multivariate/ESA-Mission1-semi-supervised/'
    dataset = ESASegLoader(data_path, win_size=128, step=64, mode='train')
    print(len(dataset))
    data, label = dataset[0]
    print(data.shape)

    print(label.shape)
