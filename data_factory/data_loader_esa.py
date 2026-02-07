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
from dateutil.parser import parse as parse_date


class ESASegLoader(Dataset):
    def __init__(self, data_path, train_length, test_length, win_size, step, mode="train", target_channels = None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load data
        df = pd.read_csv(
            f"{data_path}/{train_length}.train.csv",
            parse_dates=["timestamp"]
        )

        # Identify channels
        self.all_channels = [c for c in df.columns if c.startswith("channel_")]
        self.telecommand_cols = [c for c in df.columns if c.startswith("telecommand_")]

        if target_channels is None:
            self.target_channels = self.all_channels
        else:
            self.target_channels = [c for c in target_channels if c in self.all_channels]

        print(f"Total channels: {len(self.all_channels)}")
        print(f"Using channels: {len(self.target_channels)}")

        # -------- TIME-BASED VALIDATION SPLIT --------
        validation_date_split = df["timestamp"].max() - pd.DateOffset(months=3)
        self.validation_date_split = validation_date_split

        train_df = df[df["timestamp"] <= validation_date_split]
        val_df   = df[df["timestamp"] >  validation_date_split]

        # -------- NORMALISATION (FIT ON TRAIN ONLY) --------
        train_data = train_df[self.target_channels].values.astype(np.float32)

        self.scaler.fit(train_data)

        train_data = self.scaler.transform(train_data)
        val_data   = self.scaler.transform(
            val_df[self.target_channels].values.astype(np.float32))

        # -------- SELECT MODE --------
        if self.mode == "train":
            self.train = train_data
            print("train:", self.train.shape)

        elif self.mode == "val":
            self.val = val_data
            print("val:", self.val.shape)

        if self.mode == 'test' or self.mode == 'thre':
          # test data
          test_df = pd.read_csv(data_path + '/' + test_length + '.test.csv')
          self.label_columns = [f'is_anomaly_{ch}' for ch in self.target_channels]
          test_data = test_df[self.target_channels].values.astype(np.float32)
          test_labels = (test_df[self.label_columns].values != 0).astype(np.float32)

          timestamps = test_df['timestamp']
          timestamps = pd.to_datetime(timestamps)
        

          test_data = np.nan_to_num(test_data)

          test_data = self.scaler.transform(test_data)

          self.test = test_data
          self.test_labels = test_labels
          self.test_timestamps = timestamps

          # extract a small portion for testing to speed up experiments
          #test_len = len(test_data)
          #self.test = test_data[:int(test_len * 0.1)]
          #self.test_labels = test_labels[:int(test_len * 0.1)]
          #self.test_timestamps = timestamps[:int(test_len * 0.1)]


          print("test:", self.test.shape)
        
        # Create sliding windows
        self._create_windows()

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
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size)), index
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(np.zeros(self.win_size)) , index
            #   np.float32(self.val_labels[index:index + self.win_size]), 
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), index

    def _create_windows(self):
      """Create sliding window indices"""
      self.window_indices = []
      if self.mode == 'train':
        n_samples = len(self.train)

      elif self.mode == 'val':
        n_samples = len(self.val)
        
      else:
        n_samples = len(self.test)
      
      for i in range(0, n_samples - self.win_size + 1, self.step):
          self.window_indices.append(i)
          
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
    data_path = 'dataset/ESA'
    dataset = ESASegLoader(data_path, train_length= '1_months', test_length = '1_months', win_size=128, step=64, mode='test')
    print(len(dataset))
    data, label, index = dataset[0]
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


class ESASegLoader_clean(Dataset):
    def __init__(
        self,
        data_path,
        train_length,
        test_length,
        win_size,
        step,
        mode="train",
        target_channels=None,
        min_segment_length=260,
    ):
        self.mode = mode
        self.win_size = win_size
        self.step = step
        self.scaler = StandardScaler()
      
        # --------------------------------------------------
        # Load training data
        # --------------------------------------------------
        df = pd.read_csv(f"{data_path}/{train_length}.train.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Validation = last 3 months
        validation_date_split = df.index.max() - pd.DateOffset(months=3)
        self.validation_date_split = validation_date_split

        self.all_channels = [c for c in df.columns if c.startswith("channel_")]

        if target_channels is None:
            self.target_channels = self.all_channels
        else:
            self.target_channels = [c for c in target_channels if c in self.all_channels]

        self.label_columns = [f"is_anomaly_{c}" for c in self.target_channels]

        # --------------------------------------------------
        # Extract clean segments 
        # --------------------------------------------------
        train_segments, val_segments = self._extract_clean_segments(
            df,
            min_segment_length=min_segment_length,
        )

        # --------------------------------------------------
        # Fit scaler ONLY on training segments
        # --------------------------------------------------
        train_concat = np.vstack([seg[self.target_channels].values for seg in train_segments])
        train_concat = np.nan_to_num(train_concat)
        self.scaler.fit(train_concat)

        # --------------------------------------------------
        # Store segments (scaled)
        # --------------------------------------------------
        if self.mode == "train":
            self.segments = [
                self.scaler.transform(np.nan_to_num(seg[self.target_channels].values))
                for seg in train_segments
            ]

        elif self.mode == "val":
            self.segments = [
                self.scaler.transform(np.nan_to_num(seg[self.target_channels].values))
                for seg in val_segments
            ]

        elif self.mode in ["test", "thre"]:
            test_df = pd.read_csv(f"{data_path}/{test_length}.test.csv")
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
            test_df.set_index("timestamp", inplace=True)

            self.test_data = self.scaler.transform(
                np.nan_to_num(test_df[self.target_channels].values)
            )
            self.test_labels = (test_df[self.label_columns].values != 0).astype(np.float32)
            self.test_timestamps = test_df.index

        # --------------------------------------------------
        # Build window index (segment-aware)
        # --------------------------------------------------
        self.window_index = []

        if self.mode in ["train", "val"]:
            for seg_id, seg in enumerate(self.segments):
                T = len(seg)
                for start in range(0, T - self.win_size + 1, self.step):
                    self.window_index.append((seg_id, start))
        else:
            T = len(self.test_data)
            for start in range(0, T - self.win_size + 1, self.step):
                self.window_index.append(start)
    
    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
      
        if self.mode in ["train", "val"]:
            seg_id, start = self.window_index[idx]
            window = self.segments[seg_id][start:start + self.win_size]
            return (
                np.float32(window),
                np.zeros(self.win_size, dtype=np.float32),
                idx,
            )

        else:  # test / thre
            start = self.window_index[idx]
            return (
                np.float32(self.test_data[start:start + self.win_size]),
                np.float32(self.test_labels[start:start + self.win_size]),
                idx,
            )

    def _extract_clean_segments(self, df, min_segment_length):
        df = df.copy()
        df["is_anomaly"] = 0

        for ch in self.target_channels:
            col = f"is_anomaly_{ch}"
            df.loc[df[col] > 0, col] = 1
            df["is_anomaly"] |= df[col]

        groups = df.groupby(
            (df["is_anomaly"].shift() != df["is_anomaly"]).cumsum()
        )

        clean_segments = []

        for idxs in groups.groups.values():
            if df.loc[idxs[0], "is_anomaly"] == 0:
                seg = df.loc[idxs]
                if len(seg) >= min_segment_length:
                    clean_segments.append(seg)

        # ---- time-based split if provided
        if self.validation_date_split is not None:
            #split = parse_date(self.validation_date_split)
            split = self.validation_date_split
            train_segs, val_segs = [], []

            for seg in clean_segments:
                if seg.index[-1] < split:
                    train_segs.append(seg)
                elif seg.index[0] > split:
                    val_segs.append(seg)
                else:
                    left = seg.loc[:split]
                    right = seg.loc[split:]
                    if len(left) >= min_segment_length:
                        train_segs.append(left)
                    if len(right) >= min_segment_length:
                        val_segs.append(right)

            return train_segs, val_segs

        # ---- fallback: length split
        split_idx = int(0.8 * len(clean_segments))
        return clean_segments[:split_idx], clean_segments[split_idx:]

def get_loader_segment_clean(
    data_path,
    batch_size,
    train_length,
    test_length,
    win_size=100,
    step=100,
    mode="train",
):
    dataset = ESASegLoader_clean(
        data_path=data_path,
        train_length=train_length,
        test_length=test_length,
        win_size=win_size,
        step=step,
        mode=mode,
    )

    shuffle = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True if mode == "train" else False,
    )

