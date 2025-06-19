import csv
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from numpy.random import sample
from sklearn.impute import KNNImputer
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler

from Task2 import Task2
from test_claire.loader import JamesLoader


class Task3:
    def __init__(self):
        self.task2 = Task2()
        self.sensor_types = ["Accelerometer", "Linear Accelerometer", "Gyroscope", "Magnetometer"]
        self.dirs = ["X", "Y", "Z"]
        self.attrs = []
        self.sample_rate = 100
        for sensor_type in self.sensor_types:
            for dir in self.dirs:
                self.attrs.append(f"{dir}_{sensor_type}")

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
            for sensor_type in self.sensor_types:
                group[f'X_{sensor_type}'] = signal.filtfilt(b, a, group[f"X_{sensor_type}"])
                group[f'Y_{sensor_type}'] = signal.filtfilt(b, a, group[f"Y_{sensor_type}"])
                group[f'Z_{sensor_type}'] = signal.filtfilt(b, a, group[f"Z_{sensor_type}"])
            return group

        grouped = df.groupby(['skill_type'])
        df_filtered = grouped.apply(filter_group).reset_index(drop=True)
        # df_filtered = self.add_continuous_time(df_filtered)
        # self.task2.plot_data_over_time(df_filtered, 'X_filtered', 'Y_filtered', 'Z_filtered', "continuous_time")
        return df_filtered

    def apply_kalman_filter(self, df:pd.DataFrame) -> pd.DataFrame:
        def kalman_filter(data, process_variance=1e-2, measurement_variance=0.1):
            # Initialize the Kalman Filter for a 1D signal
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[data.iloc[0]]])  # start with first data point
            kf.F = np.array([[1]])  # state transition
            kf.H = np.array([[1]])  # measurement function
            kf.P = np.array([[1]])  # initial covariance
            kf.Q = process_variance  # process noise
            kf.R = measurement_variance  # measurement noise

            filtered = []

            for measurement in data:
                kf.predict()
                kf.update(measurement)
                filtered.append(kf.x[0, 0])
            return filtered

        # Apply Kalman filter to each axis
        for attr in self.attrs[:]:
            # plt.figure(figsize=(12, 6))
            # plt.plot(df[attr], label=f'Original {attr}', alpha=0.5)
            df[attr] = df.groupby("skill_type user".split())[attr].transform(kalman_filter)
            # plt.plot(df[attr], label=f'Filtered {attr}', linewidth=2)
            # plt.legend()
            # plt.xlabel('Sample Index')
            # plt.ylabel('X Value')
            # plt.title('Kalman Filter on X axis')
            # plt.show()
        # print(df.columns)
        return df

    def trim_data(self, df:pd.DataFrame) -> pd.DataFrame:
        def trim(sdf):
            # la_df = sdf[sdf['sensor_type'] == 'Linear Accelerometer']
            # timestamp = la_df['time'].loc[la_df["X"].idxmax()]
            # sl = 2
            # lower = timestamp - sl
            # upper = timestamp + sl
            # sdf = sdf[(sdf['time'] > lower) & (sdf['time'] < upper)].copy()
            # sdf['time'] -= lower
            # return sdf
            timestamp = sdf['time'].loc[sdf["X_Linear Accelerometer"].idxmax()]
            sl = 2
            lower = timestamp - sl
            upper = timestamp + sl
            sdf = sdf[(sdf['time'] > lower) & (sdf['time'] < upper)].copy()
            sdf['time'] -= lower
            return sdf

        # _, (sdf, *_) = zip(*df.groupby("user skill_type sample_number".split()))
        # for _, sdf in list(df.groupby("user skill_type sample_number".split()))[:1]:
        #     sdf[['time', "X_Accelerometer", 'Y_Accelerometer', "Z_Accelerometer"]].set_index('time').plot()
        #     plt.show()
        df_trim = df.groupby("user skill_type sample_number".split()).apply(trim).reset_index(drop=True)
        # _, (sdf, *_) = zip(*df_trim.groupby("user skill_type sample_number".split()))
        # for _, sdf in list(df_trim.groupby("user skill_type sample_number".split()))[:1]:
        #     sdf[['time', "X_Accelerometer", 'Y_Accelerometer', "Z_Accelerometer"]].set_index('time').plot()
        #     plt.show()
        return df_trim

    def impute_missing_data(self, trimmed_data:pd.DataFrame) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=10)
        cols = ['skill_type', 'user', 'sample_number', 'time']
        df = trimmed_data.drop(columns=cols)
        df2 = trimmed_data[cols].reset_index(drop=True)

        imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns).reset_index(drop=True)
        combined_df = pd.concat([df2, imputed], axis=1, ignore_index=True)
        combined_df.columns = trimmed_data.columns
        # print(combined_df.head().to_string())
        # print(combined_df.isnull().sum())
        return combined_df

    def compute_fft_features(self, segment, sample_rate):
        n = len(segment)
        fft_vals = np.fft.fft(segment)
        fft_freq = np.fft.fftfreq(n, d=1 / sample_rate)
        mask = fft_freq >= 0
        freqs = fft_freq[mask]
        magnitudes = np.abs(fft_vals[mask])
        # Basic features: total power, dominant frequency, max magnitude
        total_power = np.sum(magnitudes ** 2)
        dom_freq = freqs[np.argmax(magnitudes)]
        max_magnitude = np.max(magnitudes)
        return total_power, dom_freq, max_magnitude

    def apply_sliding_window2(self, df:pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
        def sliding_window(sdf):
            sdf.reset_index(drop=True, inplace=True)
            for start in range(0, len(sdf) - window_size + 1, stride):
                end = start + window_size
                window = sdf.iloc[start:end]
                mean_vals = window[self.attrs].mean()
                std_vals = window[self.attrs].std()
                for attr in self.attrs:
                    sdf.at[end-1, f"{attr}_mean"] = mean_vals[attr]
                    sdf.at[end-1, f"{attr}_std"] = std_vals[attr]

                    (sdf.at[end-1, f"{attr}_total_power"], sdf.at[end-1, f"{attr}_dom_freq"],
                     sdf.at[end-1, f"{attr}_max_magnitude"])  = self.compute_fft_features(window[attr], self.sample_rate)
            print(sdf.dropna().shape)
            return sdf

        for attr in self.attrs:
            df[f"{attr}_mean"] = np.nan
            df[f"{attr}_std"] = np.nan
            df[f"{attr}_total_power"] = np.nan
            df[f"{attr}_dom_freq"] = np.nan
            df[f"{attr}_max_magnitude"] = np.nan

        df = df.groupby("user skill_type".split()).apply(sliding_window).dropna().reset_index(drop=True)
        # _, (sdf, *_) = zip(*df.groupby("user skill_type sensor_type".split()))
        # print(df.dropna().shape)
        # print(df.head(5).to_string())
        # for grp, sdf in df.groupby("user skill_type sensor_type".split()):
        #     print(grp)
        #     print(sdf.dropna())
        return df

    def change_sample_number(self, df:pd.DataFrame) -> pd.DataFrame:
        modified_dfs = []
        unique_users = len(df["user"].unique())
        for i, (_, sdf) in enumerate(df.groupby('user')):
            print(i)
            print(unique_users)
            sdf["sample_number"] = sdf["sample_number"] + i * 11
            modified_dfs.append(sdf)
        df = pd.concat(modified_dfs, ignore_index=True)
        return df

    def normalize_data(self, agg_data: pd.DataFrame) -> pd.DataFrame:
        print(agg_data.columns)
        dropped_cols = ["skill_type", "user", "sample_number", "time"]
        df = agg_data.drop(columns=dropped_cols)
        df2 = agg_data[dropped_cols].reset_index(drop=True)
        norm = pd.DataFrame(MinMaxScaler().fit_transform(df)).reset_index(drop=True)
        combined_df = pd.concat([df2, norm], axis=1, ignore_index=True)
        combined_df.columns = agg_data.columns
        return combined_df

if __name__ == '__main__':
    loader = JamesLoader()
    task3 = Task3()

    df = loader.from_parquet()

    df = task3.impute_missing_data(df)
    df = task3.apply_kalman_filter(df)
    df = task3.trim_data(df)
    # df = task3.apply_sliding_window2(df, 100, 50)
    df = task3.change_sample_number(df)
    df = task3.normalize_data(df)
    print(df.head().to_string())
    print(df.columns)
    df.to_csv("datasets/lstm2.csv")
    # df.to_csv("datasets/applied_kalman_100_50_2.csv")

    # df = task3.trim_data(df)
    # df = task3.apply_sliding_window2(df, 100, 50)


