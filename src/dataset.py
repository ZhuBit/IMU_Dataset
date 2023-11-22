import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import random
from scipy import interpolate
import numpy as np
import re

class IMUDatasetResample(Dataset):
    def __init__(self, data_dir, freq=25, sample_len=5, ambidextral = True,
                 augmentation = False, overwrite_factor = 0.1): #freq=25, sample_len=5000,
        super(IMUDatasetResample, self).__init__()

        # List all files in the directory
        all_files = os.listdir(data_dir)

        # Filter out files that end with '_L.csv'
        # self.left_files = [f for f in all_files if f.endswith('_L_annotated.csv')]
        self.left_files = []
        for f in all_files:
            if f.endswith('_L_annotated.csv'):
                for i in range(5):
                    self.left_files.append(f)

        # Sort the files for consistency
        self.left_files.sort()

        self.data_dir = data_dir

        self.freq = freq
        self.sample_len = sample_len

        # if we want to cut r, l, b from the sting labels
        self.ambidextral = ambidextral

        self.augmentation = augmentation
        # how much data to overwrite with 0 if augmentation
        self.overwrite_factor  = overwrite_factor
    def __len__(self):
        return len(self.left_files)

    def crop_data(self, left_data, right_data): #, labels_data):
      # Get min and max timestamp from the first file
      min_time = left_data['timestamp_mills_ms'].min()
      #take smallest timstamp for max tim(files can be diffrent len???)
      max_time = min(left_data['timestamp_mills_ms'].max(), right_data['timestamp_mills_ms'].max())


      # Randomly select a start time from the first file such that there's at least self.sampl_len left
      start_time = random.uniform(min_time, max_time - self.sample_len)
      end_time = start_time + self.sample_len
      #print('Crop mae from: ', start_time, ' to: ', end_time)


      # # Tranform labels df so each row is ms and labels correspondingly
      # labels_data = self.transform_labels(labels_data)
      # # add offset of min_time IMU to time of labels, so IMU and lables have same refernce time
      # labels_data['time'] = labels_data['time'] + min_time

      # Crop both dataframes based on the start and end time
      cropped_left = left_data[(left_data['timestamp_mills_ms'] >= start_time) & (left_data['timestamp_mills_ms'] <= end_time)]
      cropped_right = right_data[(right_data['timestamp_mills_ms'] >= start_time) & (right_data['timestamp_mills_ms'] <= end_time)]
      # cropped_labels = labels_data[(labels_data['time'] >= start_time) & (labels_data['time'] < end_time)]

      return cropped_left, cropped_right #, cropped_labels

    def resample_imu_data(self, cropped_file):
        #create new timestamps for the period
        sec = self.sample_len #int(self.sample_len/1000)
        start_timestamp = cropped_file['timestamp_mills_ms'].iloc[0]
        end_timestamp = cropped_file['timestamp_mills_ms'].iloc[-1]

        timestamps = np.linspace(start_timestamp, end_timestamp, num=sec*self.freq) #np.arange(start, end*herz) / herz, sample len is in ms
        duplicate_rows = cropped_file[cropped_file.duplicated(['timestamp_mills_ms'])]
        #print('Duplicateeees: ', duplicate_rows)
        #create new placeholders for data - gyroscope (Gx, Gy, Gz) and accelerometer (Ax, Ay, Az)
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

    # def labels_parser(self, text):
    #     # Split the text by lines
    #     lines = text.strip().split("\n")
    #
    #     # Lists to store parsed data
    #     times = []
    #     labels = []
    #
    #     # Regular expression patterns
    #     timecode_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
    #     metadata_pattern = re.compile(r"\|M:([\w,]+)")
    #
    #     for line in lines:
    #         # Extract timecodes
    #         timecodes = timecode_pattern.findall(line)
    #         if len(timecodes) == 4:
    #             times.append(timecodes[3])
    #         else:
    #             times.append(None)
    #
    #         # Extract labels
    #         metadata_match = metadata_pattern.search(line)
    #         if metadata_match:
    #             labels.append(metadata_match.group(1))
    #         else:
    #             labels.append(None)
    #
    #     # Create DataFrame
    #     df = pd.DataFrame({
    #         "time": times,
    #         "labels": labels
    #     })
    #
    #     # Ensure that both times and labels are saved in the same row
    #     for idx, row in df.iterrows():
    #         if pd.isna(row["time"]) and not pd.isna(row["labels"]):
    #             if idx > 0:
    #                 df.at[idx-1, "labels"] = row["labels"]
    #                 df.drop(idx, inplace=True)
    #
    #     # Remove rows with None values
    #     df.dropna(how='any', inplace=True)
    #
    #     df.reset_index(drop=True, inplace=True)
    #
    #     df['time'] = df['time'].apply(self.timecode_to_milliseconds)
    #
    #     return df

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

                #annotations = annotations.append(annotation_holder, ignore_index=True)
                # annotations.loc[len(annotations)] = annotation_holder
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
        # data['timestamp_mills_ms'] -= data['timestamp_mills_ms'][0]
        # data['timestamp_mills_ms'] /= 1000

        # data['label'] = np.zerso
        # data['hand'] = 0

        data = data.assign(label=None)
        data = data.assign(hand=None)

        for num_row, row in enumerate(labels.iterrows()):
            label = row[1]['label']
            hand = row[1]['hand']

            data['label'] = np.where(
                ((data['timestamp_mills_ms'] >= row[1]['from']) & (data['timestamp_mills_ms'] < row[1]['to'])), label, data['label']) #,
                #data['label'])
            data['hand'] = np.where(
                ((data['timestamp_mills_ms'] >= row[1]['from']) & (data['timestamp_mills_ms'] < row[1]['to'])), hand, data['hand']) #,
                #data['hand'])

        if not class_as_string:
            for index, row in data.iterrows():
                data.loc[index, 'label'] = labels_to_idx[data.loc[index, 'label']]
                data.loc[index, 'hand'] = hands_to_idx[data.loc[index, 'hand']]

        labels = np.asarray(data['label'])
        hands = np.asarray(data['hand']) #data[data['label'] != '0']['hand']

        return labels, hands

    def reflect_imu_data(self, imu_data):
        # reflecting data so if the usere IMU sensor is upside-down
        # Z axis stays the same
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

    def __getitem__(self, idx):
        # Get the base name without the '_L.csv' part
        base_name = self.left_files[idx].split('_L_annotated.csv')[0]

        # Construct from names left and right, labels files
        left_file = os.path.join(self.data_dir, base_name + '_L_annotated.csv')
        right_file = os.path.join(self.data_dir, base_name + '_R_annotated.csv')
        labels_path = os.path.join(self.data_dir + '/labels/', base_name + '.csv')

        # read and pars lables from edl files
        left_data = pd.read_csv(left_file)
        right_data = pd.read_csv(right_file)

        # with open(labels_path, 'r') as f:
        #     labels_text = f.read()
        # labels_data = self.labels_parser(labels_text)

        # # there are duplicates somhow??????
        # left_data = left_data.drop_duplicates()
        # right_data = right_data.drop_duplicates()

        # crop data randomly
        # croped_left, croped_righ, croped_labels = self.crop_data(left_data, right_data, labels_data)

        croped_left, croped_righ = self.crop_data(left_data, right_data)

        # resample data
        #print('L File: ', left_file)
        resample_left = self.resample_imu_data(croped_left)
        #print('R File: ', right_file)
        resample_right = self.resample_imu_data(croped_righ)

        labels_file = pd.read_csv(labels_path)
        labels = self.parse_annotations(labels_file)

        sem_labels, hand_labels = self.annotate_IMU_data(resample_left, labels)

        # Apply augmentation with 50% probability if augment is True
        if self.augmentation and random.random() > 0.5:
            one_augmentation = random.random() # which augmentation to take
            if one_augmentation > 0.5:
                resample_left = self.reflect_imu_data(resample_left)
                resample_right = self.reflect_imu_data(resample_right)
            elif  one_augmentation <= 0.5:
                resample_left = self.overwrite_imu_data(resample_left, overwrite_factor=0.1)
                resample_right = self.overwrite_imu_data(resample_right, overwrite_factor=0.1)



        left_final = np.asarray(resample_left.iloc[:, 1:]).transpose() #'GX', 'GY', 'GZ', 'AX', 'AY', 'AZ'
        right_final = np.asarray(resample_right.iloc[:, 1:]).transpose()

        # Convert the DataFrames to tensors
        left_tensor = torch.tensor(left_final, dtype=torch.float32)
        right_tensor = torch.tensor(right_final, dtype=torch.float32)

        labels_tensor = torch.tensor(sem_labels.astype(float), dtype=torch.short)
        hand_tensor = torch.tensor(hand_labels.astype(float), dtype=torch.short)


        # labels_list = resample_labels['labels'].tolist()

        return left_tensor, right_tensor, labels_tensor, hand_tensor