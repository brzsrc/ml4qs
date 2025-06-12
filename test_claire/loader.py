import os
from typing import Dict, List
import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md

class JamesLoader:
    # dir = 'datasets/badminton/Claire'
    dir = 'datasets/badminton/eric2'

    def main2(self) -> pd.DataFrame:
        # Base directory path
        base_dir = 'datasets/badminton'
        all_dfs = []

        for entry in os.scandir(base_dir):
            if not entry.is_dir(): continue

            for file in os.listdir(entry):
                skill_type = file.replace(f'_{entry.name}.xls', '')
                # skill_type = file.replace('_keeley.xls', '')
                xls = pd.ExcelFile(os.path.join(entry.path, file))
                dfs = []
                for sheet_name in xls.sheet_names:
                    if sheet_name in {'Location', 'Proximity', 'Metadata Device', 'Barometer',
                                      'Metadata Time'}: continue
                    df = xls.parse(sheet_name)
                    df.columns = ['time', 'X', 'Y', 'Z']
                    df['sensor_type'] = sheet_name
                    df["time"] = abs(df["time"].round(2))
                    df.drop_duplicates(subset=['time'], inplace=True, keep='first')
                    # duplicates = df[df.duplicated()]
                    # print("Duplicate:", duplicates, "sheet name: ", sheet_name, "file name: ", file)
                    assert not df.duplicated().any()
                    dfs.append(df)

                all_df = pd.concat(dfs)
                assert not all_df.duplicated().any()
                meta = xls.parse('Metadata Time')[['event', 'experiment time']]
                meta['experiment time'] = abs(meta['experiment time'].round(2))
                m2 = pd.DataFrame(
                    dict(start=meta[meta['event'] != 'PAUSE']['experiment time'].reset_index(drop=True)))
                m2["sample_number"] = m2.index.map(int)
                all_df['sample_number'] = m2.set_index('start').asof(all_df.time).reset_index(drop=True)
                all_df['skill_type'] = skill_type
                all_df['user'] = entry.name
                all_df = all_df[['skill_type', 'user', 'sample_number', 'sensor_type', 'time', 'X', 'Y', 'Z']].copy()
                all_dfs.append(all_df)

        all_dfx = pd.concat(all_dfs)
        assert not all_dfx.duplicated().any()
        print(all_dfx.shape)
        print(all_dfx.head().to_string())
        print(all_dfx.isnull().sum())
        return all_dfx

    def convert_sensor_type(self, df:pd.DataFrame) -> pd.DataFrame:
        sensor_types = df.sensor_type.unique()
        result = None

        cols = "X Y Z".split()
        for i, sensor in enumerate(sensor_types[:]):
            # Filter for the sensor_type
            df_sensor = df[df['sensor_type'] == sensor]
            # print(df_sensor.head(100).to_string())
            df_sensor = df_sensor.drop(columns=["sensor_type"]).rename(columns={col: f"{col}_{sensor}" for col in cols})
        #     # Merge with the result DataFrame
            if i == 0:
                result = df_sensor
            else:
                result = pd.merge(result, df_sensor, on=['skill_type', 'user', 'sample_number', 'time'], how='outer')
                # print(result.shape)
                # print(result.head().to_string())
                # print(result[result["X_Linear Accelerometer"].isnull()].to_string())
        return result

    @classmethod
    def from_parquet(cls) -> pd.DataFrame:
        path = "datasets/all_data.parquet"
        df = pd.read_parquet(path)
        print("loaded!")
        return df

if __name__ == '__main__':
    loader = JamesLoader()
    df = loader.main2()
    df = loader.convert_sensor_type(df)
    # print(df.shape)
    # print(df.isnull().sum())
    df.to_parquet("datasets/all_data.parquet")
    df.to_csv("datasets/all_data.csv")
    # loader = Loader()
    # all_sheets  = loader.from_excel()
    # df = loader.split_data_by_metadata_time(all_sheets["Accelerometer"], all_sheets["Metadata Time"])
    # df = JamesLoader.from_parquet()
    # print(df.head())
