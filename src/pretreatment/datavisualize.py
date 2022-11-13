import pandas as pd
from pandas import DataFrame
import numpy as np


def show_columns(name: str, data: DataFrame):
    print(name, '\n', data.columns)


def get_data_list(data, num: int):
    d = np.array(data)
    li = d[:, num]
    return np.unique(li)


def show_group_count(data: pd.DataFrame, index):
    for i in list(np.unique(np.array(data)[:, index])):
        length = len(data[data.iloc[:, index] == i])
        print('group name: ', i, '\tdata count: ', length)
