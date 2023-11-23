import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from scipy import interpolate
import numpy as np
from torch.utils.data import DataLoader
from flirt.acc import get_acc_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

class SlidingWindowIMUsDataset(Dataset):
    def __init__(self, data_dir='data/train', window_len=20000, hop=500, sample_len=2000, augmentation=False, ambidextrous=False): # time in ms
        super(SlidingWindowIMUsDataset, self).__init__()

        # List all files in the directory
        all_files = os.listdir(data_dir)

        # Filter out files that end with '_L.csv'
        # self.left_files = [f for f in all_files if f.endswith('_L_annotated.csv')]
        self.left_files = []
        for f in all_files:
            if f.endswith('_L_annotated.csv'):
                self.left_files.append(f)

        # Sort the files for consistency
        self.left_files.sort()

        self.data_dir = data_dir
        self.window_len = window_len # How big is the main window
        self.sample_len = sample_len # How big is the sample we want to take from the window
        self.hop = hop # How much to move the window each time

        # if we want to cut r, l, b from the sting labels
        self.ambidextrous = ambidextrous

        self.augmentation = augmentation

    def __len__(self):
        return len(self.left_files)


    """def crop_data(self, left_data, right_data):
        # determine the overlapping time range
        min_time = max(left_data['timestamp_mills_ms'].min(), right_data['timestamp_mills_ms'].min())
        max_time = min(left_data['timestamp_mills_ms'].max(), right_data['timestamp_mills_ms'].max())


        if max_time - min_time < self.window_len:
            raise ValueError("Left data does not have enough data to support the window length.")

        start_time = random.uniform(min_time, max_time - self.window_len)
        end_time = start_time + self.window_len

        cropped_left = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] <= end_time)]

        right_start_time = right_data[right_data['timestamp_mills_ms'] <= start_time]['timestamp_mills_ms'].max()
        right_end_time = right_data[right_data['timestamp_mills_ms'] >= end_time]['timestamp_mills_ms'].min()

        if pd.isna(right_start_time) or pd.isna(right_end_time):
            raise ValueError("Right data does not have timestamps close enough to match the cropped left data.")

        cropped_right = right_data[(right_data['timestamp_mills_ms'] >= right_start_time) & (right_data['timestamp_mills_ms'] <= right_end_time)]

        return cropped_left, cropped_right"""

    def resample_imu_data(self, cropped_file):
        #create new timestamps for the period
        sec = self.sample_len #int(self.sample_len/1000)
        start_timestamp = cropped_file['timestamp_mills_ms'].iloc[0]
        end_timestamp = cropped_file['timestamp_mills_ms'].iloc[-1]

        timestamps = np.linspace(start_timestamp, end_timestamp, num=sec*self.freq) #np.arange(start, end*herz) / herz, sample len is in ms
        resampled_data = np.zeros((len(timestamps), 7))

        resampled_data[:, 0] = timestamps

        # drop duplicate timestamps
        cropped_file = cropped_file.drop_duplicates(subset='timestamp_mills_ms', keep="last")

        for i, j in zip(['GX', 'GY', 'GZ', 'AX', 'AY', 'AZ'], range(6)):

            #create a function that models the data
            #note - i find quadratic function to work well, cubic overfits a bit...
            #see documentation on functions:
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
            fun = interpolate.interp1d(x=cropped_file['timestamp_mills_ms'].to_numpy(),
                                       y=cropped_file[i].to_numpy(),
                                       kind='quadratic', #quadratic, cubic, etc....
                                       fill_value='extrapolate')
            #revaluate the function on the new timestamps
            resampled_data[:, j+1] = fun(timestamps)

        resampled_data = pd.DataFrame(resampled_data, columns=['timestamp_mills_ms','GX', 'GY', 'GZ', 'AX', 'AY', 'AZ'])

        return resampled_data

    def timecode_to_milliseconds(self, timecode):
        # Assuming 30 frames per second
        milliseconds_per_frame = 1000 / 30

        hours, minutes, seconds, frames = map(int, timecode.split(":"))
        total_milliseconds = (0 * 3600 + minutes * 60 + seconds) * 1000 + frames * milliseconds_per_frame

        return int(total_milliseconds)

    def transform_labels(self, df):
        """
        Transforms the input labels DataFrame to have a row for each millisecond,
        filling the 'labels' column based on the start time of each label
        in the original DataFrame until the next label starts.
        """
        # Create a new DataFrame with a range of times from 0 to the end of the "time" column
        new_df = pd.DataFrame({'time': range(0, df['time'].iloc[-1] + 1)})

        # Initialize a new "labels" column with NaN values
        new_df['labels'] = None

        # Fill the "labels" column in the new DataFrame
        for i in range(len(df) - 1):
            start_time = df['time'].iloc[i]
            end_time = df['time'].iloc[i + 1]
            new_df.loc[start_time:end_time-1, 'labels'] = df['labels'].iloc[i]

        # Fill the last label
        new_df.loc[df['time'].iloc[-1]:, 'labels'] = df['labels'].iloc[-1]

        # If we dont care for right or left label
        if self.ambidextral:
            new_df['labels'] = new_df['labels'].str.split(',|-|_').str[0]

        return new_df

    def resample_lables_data(self, df):
        step = 1000 // self.freq
        # take every step element
        # TODO how to make this function follow btter IMU data?
        labels_resampled = df.iloc[::step, :]
        return labels_resampled

    def parse_annotations(self, labels_df):
        """"
        :ivar: DF of Davici Annotations
        :return: DF with labels[label, from(sec), to(sec), duration(sec)
        """
        annotations = pd.DataFrame(columns=['label', 'hand', 'from', 'to', 'duration'])
        for i, row in labels_df.iterrows():
            if row['Notes'] != 'End' and row['Notes'] != 'end':
                raw_label = row['Notes'].split(',')
                if len(raw_label) == 1:
                    sem_class = raw_label[0]
                    hand = 'b'
                else:
                    sem_class = raw_label[0]
                    hand = raw_label[1]

                annotation_holder = {
                    'label': sem_class,
                    'hand': hand,
                    'from': self.get_sec(row['Record In'][3:]),
                    'to': self.get_sec(labels_df.iloc[i + 1]['Record In'][3:])}
                annotation_holder['duration'] = annotation_holder['to'] - annotation_holder['from']

                annotations = pd.concat([annotations, pd.DataFrame([annotation_holder])], ignore_index=True)

        return annotations

    def get_sec(self, time_str):
        m, s, mm = time_str.split(':')
        return int(m) * 60 + int(s) + float(mm) / 100

    def annotate_IMU_data(self, dataframe, labels, class_as_string=False):

        labels_to_idx = {
            'screwing': 0,
            'cordless': 1,
            'fuegen': 2,
            'fugen': 2, #typo
            'sawing': 3,
            'handling': 4,
            'handling ': 4, #typo
            'handeling': 4,  # typo in annotations
            'handlingb':4,
            'hammering': 5,
            'sorting': 6,
            'measuring ': 7,  # typo
            'measuring': 7,
            'cabeling': 8,  # typo
            'cabling': 8,
            'background': 9,
            '' : 9
            #'0': 10  # data with this label will be cut off / left out (incorrect synchronisation...)
        }

        hands_to_idx = {'l': 0, 'r': 1, 'b':2, '':0}

        data = dataframe.copy()

        data = data.assign(label=None)
        data = data.assign(hand=None)

        for num_row, row in enumerate(labels.iterrows()):
            label = row[1]['label']
            hand = row[1]['hand']

            data['label'] = np.where(
                ((data['timestamp_mills_ms'] >= row[1]['from']) & (data['timestamp_mills_ms'] < row[1]['to'])), label, data['label']) # #data['label'])
            data['hand'] = np.where(
                ((data['timestamp_mills_ms'] >= row[1]['from']) & (data['timestamp_mills_ms'] < row[1]['to'])), hand, data['hand']) # data['hand'])

        if not class_as_string:
            for index, row in data.iterrows():
                data.loc[index, 'label'] = labels_to_idx[data.loc[index, 'label']]
                data.loc[index, 'hand'] = hands_to_idx[data.loc[index, 'hand']]

        labels = np.asarray(data['label'])
        hands = np.asarray(data['hand']) #data[data['label'] != '0']['hand']

        return labels, hands

    def reflect_imu_data(self, imu_data):
        # reflecting data so if the usere IMU sensor is upside-down; Z axis stays the same
        imu_data['GX'] = imu_data['GX'] * -1
        imu_data['AX'] = imu_data['AX'] * -1

        imu_data['GY'] = imu_data['GY'] * -1
        imu_data['AY'] = imu_data['AY'] * -1

        return imu_data

    def overwrite_imu_data(self, imu_data, overwrite_factor=0.1):

        # calculate the start and end indices for masking
        total_length = len(imu_data)
        mask_length = int(total_length * overwrite_factor)

        # randomly select the starting point for masking
        start_idx = random.randint(0, total_length - mask_length)
        end_idx = start_idx + mask_length

        # mask the attributes with value 0
        attributes = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        for attr in attributes:
            imu_data.loc[start_idx:end_idx, attr] = 0

        return imu_data
    def get_sample(self, cropped_left, cropped_right):
        # Check if the timestamp range is sufficient for the sample
        time_range_left = cropped_left['timestamp_mills_ms'].iloc[-1] - cropped_left['timestamp_mills_ms'].iloc[0]
        time_range_right = cropped_right['timestamp_mills_ms'].iloc[-1] - cropped_right['timestamp_mills_ms'].iloc[0]

        if time_range_left < self.sample_len or time_range_right < self.sample_len:
            return None, None  # Not enough data

        # Find the end timestamp for the sample
        end_time_left = cropped_left['timestamp_mills_ms'].iloc[0] + self.sample_len
        end_time_right = cropped_right['timestamp_mills_ms'].iloc[0] + self.sample_len

        # Extract the samples up to the calculated end times
        sample_left = cropped_left[cropped_left['timestamp_mills_ms'] <= end_time_left]
        sample_right = cropped_right[cropped_right['timestamp_mills_ms'] <= end_time_right]

        # Calculate the mean of the 'AX' attribute for both samples
        mean_ax_left = sample_left['AX'].mean()
        mean_ax_right = sample_right['AX'].mean()

        return mean_ax_left, mean_ax_right

    def make_hop(self, cropped_left, cropped_right):
        # Calculate the new start time for both left and right data after the hop
        new_start_time_left = cropped_left['timestamp_mills_ms'].min() + self.hop
        new_start_time_right = cropped_right['timestamp_mills_ms'].min() + self.hop

        # Crop the data to start from the new start time
        new_cropped_left = cropped_left[cropped_left['timestamp_mills_ms'] >= new_start_time_left]
        new_cropped_right = cropped_right[cropped_right['timestamp_mills_ms'] >= new_start_time_right]

        return new_cropped_left, new_cropped_right

    """def make_hop(self, cropped_left, cropped_right):
        # Find the timestamp to hop to
        hop_timestamp_left = cropped_left['timestamp_mills_ms'].iloc[0] + self.hop
        hop_timestamp_right = cropped_right['timestamp_mills_ms'].iloc[0] + self.hop

        # Crop the data to only include data after the hop timestamp
        cropped_left = cropped_left[cropped_left['timestamp_mills_ms'] >= hop_timestamp_left]
        cropped_right = cropped_right[cropped_right['timestamp_mills_ms'] >= hop_timestamp_right]

        return cropped_left, cropped_right"""
    def crop_data(self, left_data, right_data):
        # Determine the overlapping time range
        min_time = max(left_data['timestamp_mills_ms'].min(), right_data['timestamp_mills_ms'].min())
        max_time = min(left_data['timestamp_mills_ms'].max(), right_data['timestamp_mills_ms'].max())

        # Check if the overlapping range is at least win_size
        if max_time - min_time < self.window_len:
            raise ValueError("Insufficient overlapping data to support the window length.")

        # find a start time such that a window is possible
        start_time = random.uniform(min_time, max_time - self.window_len)
        end_time = start_time + self.window_len

        # Crop both datasets to this initial window
        cropped_left = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] <= end_time)]
        cropped_right = right_data[(right_data['timestamp_mills_ms'] >= start_time) & (right_data['timestamp_mills_ms'] <= end_time)]

        # Check and adjust the actual range of timestamps in the cropped data
        actual_left_range = cropped_left['timestamp_mills_ms'].max() - cropped_left['timestamp_mills_ms'].min()
        actual_right_range = cropped_right['timestamp_mills_ms'].max() - cropped_right['timestamp_mills_ms'].min()

        while actual_left_range < self.window_len or actual_right_range < self.window_len:
            end_time += 1  # Increment end_time
            cropped_left = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] <= end_time)]
            cropped_right = right_data[(right_data['timestamp_mills_ms'] >= start_time) & (right_data['timestamp_mills_ms'] <= end_time)]
            actual_left_range = cropped_left['timestamp_mills_ms'].max() - cropped_left['timestamp_mills_ms'].min()
            actual_right_range = cropped_right['timestamp_mills_ms'].max() - cropped_right['timestamp_mills_ms'].min()
        print('Main crop ', cropped_left['timestamp_mills_ms'].iloc[0], cropped_left['timestamp_mills_ms'].iloc[-1])
        return cropped_left, cropped_right
    def create_segments(self, cropped_left, cropped_right):
        segments_left = []
        segments_right = []

        start_time_left = cropped_left['timestamp_mills_ms'].min()
        start_time_right = cropped_right['timestamp_mills_ms'].min()

        num_segments = ((self.window_len - self.sample_len) // self.hop) + 1

        for _ in range(num_segments):
            end_time_left = start_time_left + self.sample_len
            end_time_right = start_time_right + self.sample_len

            segment_left = cropped_left[(cropped_left['timestamp_mills_ms'] >= start_time_left) &
                                        (cropped_left['timestamp_mills_ms'] < end_time_left)]
            segment_right = cropped_right[(cropped_right['timestamp_mills_ms'] >= start_time_right) &
                                          (cropped_right['timestamp_mills_ms'] < end_time_right)]
            print('Size ', len(segment_left), len(segment_right))
            segments_left.append(segment_left)
            segments_right.append(segment_right)

            start_time_left += self.hop
            start_time_right += self.hop
        print('Segment left range in ms', segments_left[0]['timestamp_mills_ms'].iloc[-1] - segments_left[0]['timestamp_mills_ms'].iloc[0])


        return segments_left, segments_right
    """def create_segments(self, cropped_left, cropped_right):
        segments_left = []
        segments_right = []

        start_time_left = cropped_left['timestamp_mills_ms'].min()
        start_time_right = cropped_right['timestamp_mills_ms'].min()

        # equal range in ms segments
        num_segments = ((self.window_len - self.sample_len) // self.hop) + 1

        for _ in range(num_segments):
            end_time_left = start_time_left + self.sample_len  # 2000 ms
            end_time_right = start_time_right + self.sample_len  # 2000 ms

            segment_left = cropped_left[(cropped_left['timestamp_mills_ms'] >= start_time_left) &
                                        (cropped_left['timestamp_mills_ms'] < end_time_left)]
            segment_right = cropped_right[(cropped_right['timestamp_mills_ms'] >= start_time_right) &
                                          (cropped_right['timestamp_mills_ms'] < end_time_right)]

            segments_left.append(segment_left)
            segments_right.append(segment_right)

            # Increment the start times by the hop size (500 ms)
            start_time_left += self.hop
            start_time_right += self.hop
        print('Segment left range in ms', segments_left[0]['timestamp_mills_ms'].iloc[-1] - segments_left[0]['timestamp_mills_ms'].iloc[0])
        print('Size ', len(segments_left), len(segments_right))
        print(segments_left)
        return segments_left, segments_right"""
    def process_segment(self, segment):
        # claulate frequency for segment
        df = segment.copy()
        pd.set_option('display.max_columns', None)
        df.loc[:, 'timestamp_mills_ms'] = pd.to_numeric(df['timestamp_mills_ms'], errors='coerce')
        df.loc[:, 'time_diff'] = df['timestamp_mills_ms'].diff()
        average_time_diff_ms = df['time_diff'].mean()
        average_time_diff_s = average_time_diff_ms / 1000
        sampling_frequency = int(1 / average_time_diff_s)

        # calculate acc features
        flirt_acc_features = get_acc_features(segment[['AX', 'AY', 'AZ']], window_length=2,
                                        window_step_size=2, data_frequency=sampling_frequency)
        std_features = df[['AX', 'AY', 'AZ']].std()
        for col in std_features.index:
            flirt_acc_features[f'std_{col}'] = std_features[col]
    
        percentiles = [0.25, 0.5, 0.75]
        percentile_features = df[['AX', 'AY', 'AZ']].quantile(percentiles)
        for percentile in percentiles:
            for col in percentile_features.columns:
                flirt_acc_features[f'percentile_{col}_{int(percentile*100)}'] = percentile_features.loc[percentile, col]
        # calculate gyro features
        
        means = [
            segment['AX'].mean(),
            segment['AY'].mean(),
            segment['AZ'].mean(),
            segment['GX'].mean(),
            segment['GY'].mean(),
            segment['GZ'].mean()
        ]
        #print('Means', means)
        return means

    def __getitem__(self, idx):
        #print(f"Type of idx: {type(idx)}")
        # Get the base name without the '_L.csv' part
        idx = int(idx)
        base_name = self.left_files[idx].split('_L_annotated.csv')[0]
        print('Base name', base_name)

        # Construct from names left and right, labels files
        left_file = os.path.join(self.data_dir, base_name + '_L_annotated.csv')
        right_file = os.path.join(self.data_dir, base_name + '_R_annotated.csv')

        # read and pars lables from edl files
        left_data = pd.read_csv(left_file)
        right_data = pd.read_csv(right_file)
        left_data = left_data[['timestamp_mills_ms', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'label', 'hand']]
        right_data = right_data[['timestamp_mills_ms', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'label', 'hand']]

        left_data['timestamp_mills_ms'] = left_data['timestamp_mills_ms'].astype(float) * 1000
        right_data['timestamp_mills_ms'] = right_data['timestamp_mills_ms'].astype(float) * 1000

        cropped_left, cropped_right = self.crop_data(left_data, right_data)
        print('Croped left range in ms', cropped_left['timestamp_mills_ms'].iloc[-1] - cropped_left['timestamp_mills_ms'].iloc[0])
        print('Croped right range in ms', cropped_right['timestamp_mills_ms'].iloc[-1] - cropped_right['timestamp_mills_ms'].iloc[0])
        # Create segments
        segments_left, segments_right = self.create_segments(cropped_left, cropped_right)

        # Process segments as needed
        # For example, calculate features for each segment
        features_left = [self.process_segment(segment) for segment in segments_left]
        features_right = [self.process_segment(segment) for segment in segments_right]

        print('Len left', len(features_left))
        print('Len right', len(features_right))
        """features_left = []
        features_right = []
        
        while True:
            f_left, f_right = self.get_sample(cropped_left, cropped_right)
            if f_left is None or f_right is None:
                break
            features_left.append(f_left)
            features_right.append(f_right)
            cropped_left, cropped_right = self.make_hop(cropped_left, cropped_right)
        print('Len left', len(features_left))
        print('Len right', len(features_right))"""
        # Apply augmentation with 50% probability if augment is True
        if self.augmentation and random.random() > 0.5:
            one_augmentation = random.random() # which augmentation to take
            if one_augmentation > 0.5:
                pass
            elif  one_augmentation <= 0.5:
                pass

        #left_final = np.asarray(resample_left.iloc[:, 1:]).transpose() #'GX', 'GY', 'GZ', 'AX', 'AY', 'AZ'
        #right_final = np.asarray(resample_right.iloc[:, 1:]).transpose()

        # Convert the DataFrames to tensors
        #left_tensor = torch.tensor(left_final, dtype=torch.float32)
        #right_tensor = torch.tensor(right_final, dtype=torch.float32)

        #labels_tensor = torch.tensor(sem_labels.astype(float), dtype=torch.short)
        #hand_tensor = torch.tensor(hand_labels.astype(float), dtype=torch.short)

        return torch.rand(4), torch.rand(4)

if __name__ == "__main__":
    data = SlidingWindowIMUsDataset(data_dir='data/train', window_len=20000, hop=500, sample_len=2000, augmentation=False, ambidextrous=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for i, batch in enumerate(data_loader):
        print('**********************')
        if i == 1:
            pass
        else:
            break
        #print(i, batch)
