from pandas import DataFrame
import numpy as np

def show_columns(name: str, data: DataFrame):
    print(name, '\n', data.columns)


def get_data_list(data, num: int):
    d = np.array(data)
    li = d[:, num]
    return np.unique(li)
