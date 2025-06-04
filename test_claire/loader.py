from typing import Dict, List

import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md
from gensim.corpora import Dictionary


class Loader:
    EXCEL_PATH = 'datasets/test_claire/test_claire_all.xls'
    xls = pd.ExcelFile(EXCEL_PATH)

    @classmethod
    def from_excel(cls) -> Dict[str, pd.DataFrame]:
        all_sheets = {}
        for sheet_name in cls.xls.sheet_names[:]:
            if sheet_name != 'Location':
                # print(sheet_name)
                df = pd.read_excel(cls.EXCEL_PATH, sheet_name=sheet_name)
                all_sheets[sheet_name] = df
                # print(df.head(5))
        return all_sheets

    @classmethod
    def add_music_genre(cls, genres: List[str], meta_time: pd.DataFrame) -> pd.DataFrame:
        new_genres = [item for item in genres for _ in range(2)]
        meta_time['genres'] = new_genres
        return meta_time


    @classmethod
    def split_data_by_metadata_time(cls, df: pd.DataFrame, meta_time: pd.DataFrame) -> pd.DataFrame:
        time_col = "Time (s)"
        print(df.columns)
        meta_time = cls.add_music_genre(["no_music", "soft", "classical", "rap", "pop", "electronic"], meta_time)
        for group, sdf in meta_time.groupby('genres'):
            # print(group)
            # print(sdf)
            start_time = sdf.loc[sdf["event"] == "START", "experiment time"].iloc[0]
            end_time = sdf.loc[sdf["event"] == "PAUSE", "experiment time"].iloc[0]
            # print(start_time, end_time)
            # print(df[time_col])

            mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)
            df.loc[mask, 'genre'] = group

            df.loc[mask, "timestamp"] = df.loc[mask, time_col] - start_time

        # print(df.to_string())
        return df



if __name__ == '__main__':
    loader = Loader()
    all_sheets  = loader.from_excel()
    df = loader.split_data_by_metadata_time(all_sheets["Accelerometer"], all_sheets["Metadata Time"])