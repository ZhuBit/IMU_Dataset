import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

class SlidingWindowIMUsDataset(Dataset):
    def __init__(self, data_dir='data/train', window_len=20000, hop=500, kernel=2000):
        self.data_dir = data_dir
        self.window_len = window_len
        self.hop = hop
        self.kernel = kernel

        all_files = os.listdir(data_dir)
        self.left_files = [f for f in all_files if f.endswith('_L_annotated.csv')]
        self.right_files = [f.replace('_L_', '_R_') for f in self.left_files]
        self.label_files = [f.split('_L_')[0] + '.csv' for f in self.left_files]

        self.left_files.sort()
        self.right_files.sort()
        self.label_files.sort()

    def __len__(self):
        return len(self.left_files)

    def __getitem__(self, idx):
        left_data = pd.read_csv(os.path.join(self.data_dir, self.left_files[idx]))
        right_data = pd.read_csv(os.path.join(self.data_dir, self.right_files[idx]))

        left_data['timestamp_mills_ms'] = (left_data['timestamp_mills_ms'] * 1000).astype(int)
        right_data['timestamp_mills_ms'] = (right_data['timestamp_mills_ms'] * 1000).astype(int)

        max_start = min(left_data['timestamp_mills_ms'].max(), right_data['timestamp_mills_ms'].max()) - self.window_len
        max_start = int(max_start)  # Convert to integer

        if max_start <= 0:
            print(f"Window length {self.window_len} is too large for the dataset.")
            return torch.rand(4)

        start_time = random.randint(0, max_start)
        end_time = start_time + self.window_len

        window_left = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] <= end_time)]
        window_right = right_data[(right_data['timestamp_mills_ms'] >= start_time) & (right_data['timestamp_mills_ms'] <= end_time)]

        current_start = start_time
        while current_start + self.kernel <= end_time:
            kernel_left = window_left[(window_left['timestamp_mills_ms'] >= current_start) & (window_left['timestamp_mills_ms'] < current_start + self.kernel)]
            kernel_right = window_right[(window_right['timestamp_mills_ms'] >= current_start) & (window_right['timestamp_mills_ms'] < current_start + self.kernel)]


            print(f"Kernel from {current_start} to {current_start + self.kernel}")


            current_start += self.hop

        # Return dummy data for now
        return torch.rand(4)


dataset = SlidingWindowIMUsDataset()
print(len(dataset))
print(dataset[0])

#%%
