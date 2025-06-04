import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md

EXCEL_PATH = 'datasets/test_claire/test_claire_all.xls'
xls = pd.ExcelFile(EXCEL_PATH)


# Load each sheet into a dictionary
all_sheets = {}
for sheet_name in xls.sheet_names[:]:
    if sheet_name != 'Location':
        print(sheet_name)
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
        all_sheets[sheet_name] = df
        print(df.head(5))