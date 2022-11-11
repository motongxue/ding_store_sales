"""
@PROJECT: StoreSales - dataload.py
@IDE: PyCharm
@DATE: 2022/11/9 下午1:09
@AUTHOR: lxx
"""

import pandas as pd
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

from src.pretreatment.datamachine import format_data, fixna
from src.pretreatment.datavisualize import show_columns, get_data_list


def data_clean(data: EasyDict):
    print('n')


def data_load(config: EasyDict):
    """

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
    train = pd.read_csv(config.load.train_path + 'train.csv')
    transaction = pd.read_csv(config.load.train_path + 'transactions.csv')
    extra_holidays = pd.read_csv(config.load.extra_path + 'holidays_events.csv')
    extra_oil = pd.read_csv(config.load.extra_path + 'oil.csv')
    extra_store_inf = pd.read_csv(config.load.extra_path + 'stores.csv')
    test = pd.read_csv(config.load.test_path + 'test.csv')

    format_train, format_test = format_data(train, transaction, extra_holidays, extra_oil, extra_store_inf, test)
    format_train = torch.tensor(format_train.iloc[:, 2:].values.astype(float))
    format_test = torch.tensor(format_test.iloc[:, 2:].values.astype(float))

    format_train = DataLoader(format_train, batch_size=64, shuffle=True)
    format_test = DataLoader(format_test, batch_size=64, shuffle=True)
    return format_train, format_test

