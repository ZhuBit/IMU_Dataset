import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
class SlidingWindowIMUsDataset(Dataset):
    def __init__(self, data_dir, window_len, hop, kernel):
        self.data_dir = data_dir
        self.window_len = window_len
        self.hop = hop
        self.kernel = kernel

        # List and filter files
        all_files = os.listdir(data_dir)
        self.left_files = [f for f in all_files if f.endswith('_L_annotated.csv')]
        self.left_files.sort()

    def __len__(self):
        return len(self.left_files)

    def __getitem__(self, idx):
        # Get file names
        base_name = self.left_files[idx].split('_L_annotated.csv')[0]
        left_file = os.path.join(self.data_dir, base_name + '_L_annotated.csv')
        right_file = os.path.join(self.data_dir, base_name + '_R_annotated.csv')

        # Read data
        left_data = pd.read_csv(left_file)
        right_data = pd.read_csv(right_file)

        # Randomly select a window
        max_start = max(left_data['timestamp_mills_ms'].max(), right_data['timestamp_mills_ms'].max()) - self.window_len
        start_time = random.randint(0, max_start)
        end_time = start_time + self.window_len

        # Filter data for the selected window
        left_window = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] < end_time)]
        right_window = right_data[(right_data['timestamp_mills_ms'] >= start_time) & (right_data['timestamp_mills_ms'] < end_time)]

        # Iterate over the window in steps of self.hop
        for start in range(start_time, end_time - self.kernel, self.hop):
            end = start + self.kernel
            left_segment = left_window[(left_window['timestamp_mills_ms'] >= start) & (left_window['timestamp_mills_ms'] < end)]
            right_segment = right_window[(right_window['timestamp_mills_ms'] >= start) & (right_window['timestamp_mills_ms'] < end)]

            # Convert to tensors
            left_tensor = torch.tensor(left_segment.values, dtype=torch.float32)
            right_tensor = torch.tensor(right_segment.values, dtype=torch.float32)

            yield left_tensor, right_tensor

# Example usage
if __name__ == "__main__":
    data_dir = 'data/train'
    val_dir = './data/validate/'

    window_len = 20000 # ms
    hop = 500    # ms
    kernel = 2000 # ms
    batch_size = 32

    train_dataset = SlidingWindowIMUsDataset(data_dir, window_len, hop, kernel)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SlidingWindowIMUsDataset(data_dir, window_len, hop, kernel)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for i, (left_data, right_data) in enumerate(train_dataloader):
        print(f"Batch {i}:")
        print("Left Data Batch Shape:", left_data.shape)
        print("Right Data Batch Shape:", right_data.shape)
        # Add a break to avoid printing too much data
        if i == 0:
            break
