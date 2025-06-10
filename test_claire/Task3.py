import csv
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from numpy.random import sample

from Task2 import Task2
from test_claire.loader import JamesLoader


class Task3:
    def __init__(self):
        self.task2 = Task2()

    def add_continuous_time_old(self, df: pd.DataFrame) -> pd.DataFrame:
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
        self.task2.plot_data_over_time(df, 'X', 'Y', 'Z', "continuous_time")
        return df

    def add_continuous_time(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return df

    def apply_lowpass_filter(self, df:pd.DataFrame) -> pd.DataFrame:
        # duplicates = len(df.index[df.index.duplicated()].tolist())
        # print("Duplicate indices:", duplicates)
        # df = df.reset_index(drop=True)

        fs = 100  # Hz (set according to your data collection rate)

        # Define cutoff frequency for the lowpass filter
        cutoff = 2.0  # Hz, adjust as needed

        # Filter design parameters
        order = 4
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Design Butterworth lowpass filter
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

        def filter_group(group):
            # group['X_filtered'] = signal.filtfilt(b, a, group["X"])
            # group['Y_filtered'] = signal.filtfilt(b, a, group["Y"])
            # group['Z_filtered'] = signal.filtfilt(b, a, group["Z"])
            group['X'] = signal.filtfilt(b, a, group["X"])
            group['Y'] = signal.filtfilt(b, a, group["Y"])
            group['Z'] = signal.filtfilt(b, a, group["Z"])
            return group

        grouped = df.groupby(['skill_type', 'sensor_type'])
        df_filtered = grouped.apply(filter_group).reset_index(drop=True)
        # df_filtered = self.add_continuous_time(df_filtered)
        # self.task2.plot_data_over_time(df_filtered, 'X_filtered', 'Y_filtered', 'Z_filtered', "continuous_time")
        return df_filtered

    def trim_data(self, df:pd.DataFrame) -> pd.DataFrame:
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
        return df_trim

    def apply_sliding_window(self, df:pd.DataFrame) -> pd.DataFrame:
        attrs = ["X", "Y", "Z"]
        def sliding_window(sdf):
            sdf.reset_index(drop=True, inplace=True)
            window_size = 100
            stride = 50
            for start in range(0, len(sdf) - window_size + 1, stride):
                end = start + window_size
                window = sdf[attrs].iloc[start:end]
                mean_vals = window.mean()
                std_vals = window.std()
                for attr in attrs:
                    sdf.at[end-1, f"{attr}_mean"] = mean_vals[attr]
                    sdf.at[end-1, f"{attr}_std"] = std_vals[attr]
            return sdf

        # for attr in attrs:
        #     df[f"{attr}_mean"] = np.nan
        #     df[f"{attr}_std"] = np.nan

        df = df.groupby("user skill_type sensor_type".split()).apply(sliding_window).dropna().reset_index(drop=True)
        # _, (sdf, *_) = zip(*df.groupby("user skill_type sensor_type".split()))
        # print(df.dropna().shape)
        # print(df.head(5).to_string())
        # for grp, sdf in df.groupby("user skill_type sensor_type".split()):
        #     print(grp)
        #     print(sdf.dropna())
        return df

    def apply_sensor_type(self, df:pd.DataFrame) -> pd.DataFrame:
        sensor_types = df.sensor_type.unique()
        result = None
        df["time"] = df["time"].round(2)

        cols = "X Y Z X_mean X_std Y_mean Y_std Z_mean Z_std".split()
        for i, sensor in enumerate(sensor_types[:]):
            # Filter for the sensor_type
            df_sensor = df[df['sensor_type'] == sensor]
            # print(df_sensor.head(100).to_string())
            df_sensor = df_sensor.drop(columns=["sensor_type"]).rename(columns={col: f"{col}_{sensor}" for col in cols})
        #     # Merge with the result DataFrame
            if i == 0:
                result = df_sensor
            else:
                result = pd.merge(result, df_sensor, on=['skill_type', 'user', 'sample_number', 'time'], how='inner')
        return result


if __name__ == '__main__':
    loader = JamesLoader()
    # self.df = loader.from_parquet()
    df = loader.main2()
    df.dropna(inplace=True)

    task3 = Task3()
    # task3.add_continuous_time_old()
    df = task3.trim_data(df)
    df = task3.apply_lowpass_filter(df)
    df = task3.apply_sliding_window(df)
    df = task3.apply_sensor_type(df)
    print(df.head(10).to_string())
    df.to_csv("datasets/add_lowpass_filter.csv")
    # df = pd.read_csv("datasets/first_try.csv")


