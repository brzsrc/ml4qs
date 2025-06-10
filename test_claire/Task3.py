from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from numpy.random import sample

from Task2 import Task2
from test_claire.loader import JamesLoader


class Task3:
    def __init__(self):
        loader = JamesLoader()
        self.df = loader.from_parquet()
        self.df.dropna(inplace=True)

    def add_continuous_time_old(self):
        df = self.df.copy()
        df = df.sort_values(by=['user', 'time', "skill_type"])
        users = df["user"].unique()
        skill_types = df["skill_type"].unique()

        # Create a column for continuous time
        df['continuous_time'] = df['time']

        # Calculate offsets for each user
        user_offsets = {}
        for skill_type in skill_types:
            previous_end_time = 0
            for user in users:
                group = df[(df['user'] == user) & (df['skill_type'] == skill_type)]
                user_skill = (user, skill_type)
                if user_skill not in user_offsets:
                    user_offsets[user_skill] = previous_end_time
                last_time = group['time'].max()
                previous_end_time += last_time


        # Apply the offset to each user's time
        for (user, skill), offset in user_offsets.items():
            df.loc[(df['user'] == user) & (df['skill_type'] == skill), 'continuous_time'] += offset
        df = df[['skill_type', 'user', 'sample_number', 'sensor_type', 'time', 'continuous_time', 'X', 'Y', 'Z']].copy()
        # df.to_csv("datasets/continuous_time.csv")
        self.df = df.copy()

    def add_continuous_time(self):
        df = self.df.copy()
        df = df.sort_values(by=['user', 'time', "skill_type"])
        users = df["user"].unique()
        skill_types = df["skill_type"].unique()

        # Create a column for continuous time
        df['continuous_time'] = df['time']

        # Calculate offsets for each user
        user_offsets = {}
        for skill_type in skill_types:
            previous_end_time = 0
            for user in users:
                group = df[(df['user'] == user) & (df['skill_type'] == skill_type)]
                for grp, sdf in group.groupby(['sample_number']):
                    user_skill = (user, skill_type, grp)
                    if user_skill not in user_offsets:
                        user_offsets[user_skill] = previous_end_time
                    last_time = group['time'].max()
                    previous_end_time += last_time


        # Apply the offset to each user's time
        for (user, skill, grp), offset in user_offsets.items():
            df.loc[(df['user'] == user) & (df['skill_type'] == skill) & (df["sample_number"] == grp), 'continuous_time'] += offset
        df = df[['skill_type', 'user', 'sample_number', 'sensor_type', 'time', 'continuous_time', 'X', 'Y', 'Z']].copy()
        # df.to_csv("datasets/continuous_time.csv")
        self.df = df.copy()


    def apply_lowpass_filter(self):
        df = self.df.copy()
        # duplicates = len(df.index[df.index.duplicated()].tolist())
        # print("Duplicate indices:", duplicates)
        df = df.reset_index(drop=True)

        fs = 100  # Hz (set according to your data collection rate)

        # Define cutoff frequency for the lowpass filter
        cutoff = 2.0  # Hz, adjust as needed

        # Filter design parameters
        order = 4
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Design Butterworth lowpass filter
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        def filter_group(group):
            group['X_filtered'] = signal.filtfilt(b, a, group["X"])
            group['Y_filtered'] = signal.filtfilt(b, a, group["Y"])
            group['Z_filtered'] = signal.filtfilt(b, a, group["Z"])
            return group

        grouped = df.groupby(['skill_type', 'sensor_type'])
        df_filtered = grouped.apply(filter_group).reset_index(drop=True)
        task2 = Task2()
        # task2.plot_data_over_time(df_filtered, 'X', 'Y', 'Z', "continuous_time")
        task2.plot_data_over_time(df_filtered, 'X_filtered', 'Y_filtered', 'Z_filtered', "continuous_time")
        self.df = df_filtered.copy()

    def trim_data(self):
        df = self.df.copy()

        def trim(sdf):
            la_df = sdf[sdf['sensor_type'] == 'Linear Accelerometer']
            timestamp = la_df['time'].loc[la_df["X"].idxmax()]
            sl = 2
            lower = timestamp - sl
            upper = timestamp + sl
            sdf = sdf[(sdf['time'] > lower) & (sdf['time'] < upper)].copy()
            sdf['time'] -= lower
            return sdf
        # _, (sdf, *_) = zip(*df.groupby("user skill_type sample_number sensor_type".split()))
        # for _, sdf in list(df.groupby("user skill_type sample_number sensor_type".split()))[:3]:
        #     sdf[['time', "X", 'Y', "Z"]].set_index('time').plot()
        #     plt.show()
        df_trim = df.groupby("user skill_type sample_number".split()).apply(trim).reset_index(drop=True)
        # _, (sdf, *_) = zip(*df_trim.groupby("user skill_type sample_number sensor_type".split()))
        # for _, sdf in list(df_trim.groupby("user skill_type sample_number sensor_type".split()))[:3]:
        #     sdf[['time', "X", 'Y', "Z"]].set_index('time').plot()
        #     plt.show()
        self.df = df_trim
        print(self.df.shape)

    def apply_sliding_window(self):
        df = self.df.copy()
        window_size = 100
        df.groupby("user skill_type sensor_type".split())

if __name__ == '__main__':
    task3 = Task3()
    # task3.add_continuous_time_old()
    task3.trim_data()
    task3.add_continuous_time()
    task3.apply_lowpass_filter()
    # task3.apply_sliding_window()
