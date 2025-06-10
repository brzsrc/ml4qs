from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

from test_claire.loader import JamesLoader


class Task2:
    def __init__(self):
        loader = JamesLoader()
        self.df = loader.from_parquet()

    def nan_vals(self):
        df = self.df.copy()
        print(df[df["sample_number"].isnull()].to_string())


    def text(self):
        df = self.df.copy()
        print(df.isnull().sum().to_string())
        print(len(df))
        df.dropna(inplace=True)
        print(df.head(5).to_string())
        print(df["sensor_type"].unique())
        print(df.drop(columns="time").groupby(["sensor_type", "skill_type"]).agg(["min", "max", "mean", "median"]).to_string())

    def box_plot(self):
        df = self.df.copy()
        df.dropna(inplace=True)
        skill_types = df['skill_type'].unique()
        attrs = ["X", "Y", "Z"]
        for grp, sdf in df.groupby(["sensor_type"]):
                fig, axs = plt.subplots(nrows=len(skill_types), ncols=1, figsize=(20, len(skill_types) * 4))
                for ax, skill_type in zip(axs, skill_types):
                    for attr in attrs:
                        group = sdf[sdf['skill_type'] == skill_type]
                        ax.boxplot(group[attr])
                        ax.set_title(f'skill_type: {skill_type}')
                plt.tight_layout()
                plt.show()


    def plot_data_over_time(self, df: pd.DataFrame, x:str, y:str, z:str, time:str):
        df.dropna(inplace=True)
        skill_types = df['skill_type'].unique()
        users = df['user'].unique()
        colors = ['y', 'orange', 'purple', 'cyan', 'magenta', 'brown']
        nrows = len(skill_types)
        for grp, sdf in df.groupby('sensor_type'):
            # Plot each genre in a separate subplot
            fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(20, len(skill_types) * 4))
            if nrows == 1:
                axs = [axs]
            for ax, skill_type in zip(axs, skill_types):
                s_legend = 0
                for ui, user in enumerate(users):
                    group = sdf[(sdf['user'] == user) & (sdf['skill_type'] == skill_type)]
                    sample_numbers = group['sample_number'].unique()
                    for sample_number in sample_numbers:
                        sample_data = group[group['sample_number'] == sample_number]
                        ax.plot(sample_data[time], sample_data[x], label=(None, 'X')[not s_legend], color='b')
                        ax.plot(sample_data[time], sample_data[y], label=(None, 'Y')[not s_legend], color='g', linestyle='--')
                        ax.plot(sample_data[time], sample_data[z], label=(None, 'Z')[not s_legend], color='r', linestyle=':')
                        # Add a vertical line at the start of each sample
                        ax.axvline(x=sample_data[time].iloc[0], label = (None, user)[not sample_number], color=colors[ui], linestyle='--', linewidth=0.8)
                        s_legend += 1

                ax.set_title(f'skill_type: {skill_type}')
                ax.set_ylabel(grp)
                # ax.set_ylim(-20, 50)  # apply same y limits
                ax.legend()
            axs[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()

    def plot_data_over_time2(self, df: pd.DataFrame, x:str, y:str, z:str, time:str):
        df.dropna(inplace=True)
        print(df.head(5).to_string())
        skill_types = df['skill_type'].unique()
        users = df['user'].unique()
        print(skill_types)
        nrows = len(users) * len(skill_types)
        for grp, sdf in df.groupby(['sensor_type']):
            # Plot each genre in a separate subplot
            fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(20, nrows * 4))
            i = 0
            for skill_type in skill_types:
                for ui, user in enumerate(users):
                    ax = axs[i]
                    group = sdf[(sdf['user'] == user) & (sdf['skill_type'] == skill_type)]
                    sample_numbers = group['sample_number'].unique()
                    for sample_number in sample_numbers:

                        sample_data = group[group['sample_number'] == sample_number]
                        ax.plot(sample_data[time], sample_data[x], label=(None, 'X')[not ui], color='b')
                        ax.plot(sample_data[time], sample_data[y], label=(None, 'Y')[not ui], color='g', linestyle='--')
                        ax.plot(sample_data[time], sample_data[z], label=(None, 'Z')[not ui], color='r', linestyle=':')
                        # Add a vertical line at the start of each sample
                        ax.axvline(x=sample_data[time].iloc[0], color='y', linestyle='--', linewidth=0.8)

                    ax.set_title(f'user: {user}')
                    ax.set_ylabel(skill_type)
                    # ax.set_ylim(-20, 50)  # apply same y limits
                    ax.grid(False)
                    i = i + 1
            ax.legend()
            axs[-1].set_xlabel(f'Time (s), Sensor_type: {grp}')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    loader = JamesLoader()
    df = loader.main2()
    task = Task2()
    # task.text()
    # task.nan_vals()
    # task.plot_data_over_time2(df, "X", "Y", "Z")
    # task.box_plot()


