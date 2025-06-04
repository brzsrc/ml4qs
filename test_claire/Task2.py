from typing import Dict, List
import pandas as pd
from loader import Loader
import matplotlib.pyplot as plt

class Task2:
    def __init__(self):
        loader = Loader()
        self.all_sheets = loader.from_excel()
        self.df = loader.split_data_by_metadata_time(self.all_sheets["Gyroscope"], self.all_sheets["Metadata Time"])

    def text(self):
        df = self.df.copy()
        df.dropna(inplace=True)
        for group, sdf in df.groupby("genre"):
            # sdf.describe()
            print(sdf.describe())

    def plot_data_over_time(self):
        df = self.df.copy()
        df.dropna(inplace=True)
        print(df)

        timestamp = "timestamp"
        attrs = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']
        attrs = ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)']

        genres = df['genre'].unique()
        fig, axs = plt.subplots(nrows=len(genres), ncols=1, figsize=(40, len(genres) * 4))

        # Plot each genre in a separate subplot
        for ax, genre in zip(axs, genres):
            group = df[df['genre'] == genre]
            ax.plot(group[timestamp], group['X (rad/s)'], label='X', color='b')
            ax.plot(group[timestamp], group['Y (rad/s)'], label='Y', color='g', linestyle='--')
            ax.plot(group[timestamp], group['Z (rad/s)'], label='Z', color='r', linestyle=':')
            ax.set_title(f'Genre: {genre}')
            ax.set_ylabel('Acceleration (m/s^2)')
            # ax.set_ylim(-20, 50)  # apply same y limits
            ax.legend()
            ax.grid(True)

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        # # Group by genre
        # for genre, group in df.groupby('genre'):
        #     for attr in attrs:
        #     # for genre, group in df.groupby('genre'):
        #         # plt.plot(group[timestamp], group['X (m/s^2)'], label=f'X - {genre}')
        #         # plt.plot(group[timestamp], group['Y (m/s^2)'], label=f'Y - {genre}', linestyle='--')
        #         # plt.plot(group[timestamp], group['Z (m/s^2)'], label=f'Z - {genre}', linestyle=':')
        #         plt.plot(group[timestamp], group[attr], label=f'{attr} - {genre}', linestyle=':')
        #
        #     # Additional plot settings
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Acceleration (m/s^2)')
        #     plt.title(f'{attrs} over Time for Each Genre')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     # Show plot
        #     plt.show()



if __name__ == '__main__':
    task = Task2()
    # task.text()
    task.plot_data_over_time()


