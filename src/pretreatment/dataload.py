"""
@PROJECT: StoreSales - dataload.py
@IDE: PyCharm
@DATE: 2022/11/9 下午1:09
@AUTHOR: lxx
"""

import torch
import numpy as np
import pandas as pd
from joblib import dump
from src.pretreatment.datamachine import format_data, scale
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def data_clean(data: EasyDict):
    print('n')


def data_load(config: EasyDict, is_train: bool):
    """

    :param is_train:
    :param config:
    :return:
    """

    """
    train = pd.read_csv('./src/dataset/train/train.csv')
    transaction = pd.read_csv('./src/dataset/train/transactions.csv')
    extra_holidays = pd.read_csv('./src/dataset/extra/holidays_events.csv')
    extra_oil = pd.read_csv('./src/dataset/extra/oil.csv')
    extra_store_inf = pd.read_csv('./src/dataset/extra/stores.csv')
    test = pd.read_csv('./src/dataset/test/test.csv')

    """
    transaction = pd.read_csv(config.load.train_path + 'transactions.csv')
    extra_holidays = pd.read_csv(config.load.extra_path + 'holidays_events.csv')
    extra_oil = pd.read_csv(config.load.extra_path + 'oil.csv')
    extra_store_inf = pd.read_csv(config.load.extra_path + 'stores.csv')
    if is_train:
        dataset = pd.read_csv(config.load.train_path + 'train.csv')
    else:
        dataset = pd.read_csv(config.load.test_path + 'test.csv')
        dataset.insert(loc=4, column='sales', value=0)
    # processing dataset
    format_dataset = format_data(dataset, transaction, extra_holidays, extra_oil, extra_store_inf)
    return format_dataset


class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config: EasyDict, is_train: bool):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """

        # load raw data file
        self.dl = data_load(config, is_train)
        self.transform = MinMaxScaler()
        self.dl_input = self.dl[['store_nbr', 'family', 'onpromotion', 'events', 'dcoilwtico']]
        self.dl_target = self.dl[['sales']]
        self.length = config.train.training_length

    def __len__(self):
        # return number of store
        return int(len(self.dl) / self.length) - 2
        # return len(self.dl)

    # Will pull an index between 0 and __len__.
    def __getitem__(self, idx):
        # _input = torch.tensor(self.dl_input.iloc[idx].values)
        # # _input = _input.reshape([25])
        # target = torch.tensor(self.dl_target.iloc[idx].values)
        # return _input, target
        _input = torch.tensor(self.dl_input.iloc[idx * self.length: (idx + 1) * self.length].values)  # [10, 5]
        # _input = _input.reshape([25])
        target = torch.tensor(self.dl_target.iloc[idx * self.length: (idx + 1) * self.length].values)  # [10, 1]
        return scale(_input), scale(target)
