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
                    dfs.append(df)

                all_df = pd.concat(dfs)
                assert not all_df.duplicated().any()
                meta = xls.parse('Metadata Time')[['event', 'experiment time']]
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
        return all_dfx

    @classmethod
    def from_parquet(cls) -> pd.DataFrame:
        path = "datasets/badminton/all_data.parquet"
        df = pd.read_parquet(path)
        print("loaded!")
        return df

if __name__ == '__main__':
    df = JamesLoader().main2()
    # print(df.shape)
    df.to_parquet("datasets/badminton/all_data.parquet")
    # loader = Loader()
    # all_sheets  = loader.from_excel()
    # df = loader.split_data_by_metadata_time(all_sheets["Accelerometer"], all_sheets["Metadata Time"])
    # df = JamesLoader.from_parquet()
    # print(df.head())
