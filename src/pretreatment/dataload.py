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
from src.pretreatment.datamachine import format_data
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

    def __init__(self, config: EasyDict, is_train: bool, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """

        # load raw data file
        self.dl = data_load(config, is_train)
        self.transform = MinMaxScaler()
        # T与S分别表示用于测试与验证的数据长度，这会从数据最后部分向前截取
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # return number of store
        return len(self.dl.groupby(by=["store_nbr"]))

    # Will pull an index between 0 and __len__.
    def __getitem__(self, idx):
        # Sensors are indexed from 1
        idx = idx + 1
        # np.random.seed(0)

        start = np.random.randint(0, len(self.dl[self.dl["store_nbr"] == idx]) - self.T - self.S)
        store_number = str(idx)
        _input = torch.tensor(self.dl[self.dl["store_nbr"] == idx]
                              .iloc[start: start + self.T, 2:].values)
        target = torch.tensor(self.dl[self.dl["store_nbr"] == idx]
                              .iloc[start + self.T: start + self.T + self.S, 2:].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:, 0].unsqueeze(-1))
        # _input[:, 0] = torch.tensor(scaler.transform(_input[:, 0].unsqueeze(-1)).squeeze(-1))
        # target[:, 0] = torch.tensor(scaler.transform(target[:, 0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')
        return _input, target, store_number
