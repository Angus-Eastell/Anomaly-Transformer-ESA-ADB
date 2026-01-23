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

class ESASegLoader(Dataset):
    def __init__(self, data_path, train_length, test_length, win_size, step, mode="train", target_channels = None):
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

        if self.mode == 'train':
          #self.train = data[:int(data_len * 0.8)]
          #self.train = data

          # extract a small portion for training to speed up experiments
          self.train = data[:int(data_len * 0.1)]
          self.labels = self.labels[:int(data_len * 0.1)]
          print("train:", self.train.shape)

        if self.mode == 'val':
          self.val = data[int(data_len * 0.8):]
          self.val_labels = self.labels[int(data_len * 0.8):]
          print("val:", self.val.shape)

        if self.mode == 'test' or self.mode == 'thre':
          # test data
          test_df = pd.read_csv(data_path + '/' + test_length + '.test.csv')

          test_data = test_df[self.target_channels].values.astype(np.float32)
          test_labels = test_df[self.label_columns].values.astype(np.float32)
          timestamps = test_df['timestamp']
          timestamps = pd.to_datetime(timestamps)
        

          test_data = np.nan_to_num(test_data)

          test_data = self.scaler.transform(test_data)

          #self.test = test_data
          #self.test_label = test_labels
          #self.test_timestamps = timestamps

          # extract a small portion for testing to speed up experiments
          test_len = len(test_data)
          self.test = test_data[:int(test_len * 0.1)]
          self.test_labels = test_labels[:int(test_len * 0.1)]
          self.test_timestamps = timestamps[:int(test_len * 0.1)]


          print("test:", self.test.shape)

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
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[0:self.win_size])
            #   np.float32(self.val_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

    def get_timestamps(self):
      """Get all timestamps"""
      return self.test_timestamps

def get_loader_segment(data_path, batch_size, train_length, test_length, win_size= 100, step = 100, mode = 'train', dataset = '3_months'):

    dataset = ESASegLoader(data_path, train_length, test_length, win_size, step, mode=mode)

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

class ESALabelsParser:
    """
    Parse ESA labels.csv file for event-based evaluation
    """
    
    def __init__(self, labels_csv_path):
        """
        Args:
            labels_csv_path: Path to labels.csv file
        """
        self.labels_df = pd.read_csv(labels_csv_path)
        
        # Parse timestamps
        self.labels_df['StartTime'] = pd.to_datetime(self.labels_df['StartTime'])
        self.labels_df['EndTime'] = pd.to_datetime(self.labels_df['EndTime'])
        
        print(f"Loaded {len(self.labels_df)} anomaly events")
        print(f"Unique anomaly IDs: {self.labels_df['ID'].nunique()}")
        
    def get_labels_dataframe(self, channel_filter=None):
        """
        Get labels DataFrame, optionally filtered by channels
        
        Args:
            channel_filter: List of channel names to include, or None for all
            
        Returns:
            pandas DataFrame in ESA format
        """
        if channel_filter is None:
            return self.labels_df.copy()
        
        # Filter by channels
        filtered_df = self.labels_df[
            self.labels_df['Channel'].isin(channel_filter)
        ].copy()
        
        return filtered_df
    
    def get_full_range(self):
        """Get full time range of data"""
        return (
            self.labels_df['StartTime'].min(),
            self.labels_df['EndTime'].max()
        )
    
    def get_anomaly_categories(self):
        """Get unique anomaly categories"""
        return self.labels_df['Category'].unique()

