from math import isnan

import pandas as pd
from easydict import EasyDict

from src.pretreatment.datamachine import format_data
from src.pretreatment.datavisualize import show_columns, get_data_list


def data_clean(data: EasyDict):
    print('n')


def load(config: EasyDict):
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

    print(get_data_list(extra_store_inf, 1))
    print(get_data_list(extra_store_inf, 2))
    print(get_data_list(extra_holidays, 3))

    # combine train data and test data
    train_len = train.shape[0]
    test.insert(loc=4, column='sales', value=0)
    # combine = train.append(test, ignore_index=True)
    # train_data = train.groupby(['date']).agg({'sales': 'mean', 'onpromotion': 'mean'})

    # 清理商店无用信息，合并得到带有位置信息的训练集
    valid_extra_store_inf = extra_store_inf.drop(axis=1, columns=['type', 'cluster'])
    train_state = pd.merge(train, valid_extra_store_inf, how='left', left_on='store_nbr', right_on='store_nbr')

    # 清理无效假日，删除无用数据
    valid_holiday = extra_holidays.drop(extra_holidays[extra_holidays['transferred']].index)
    valid_holiday.drop(axis=1, columns=['locale', 'description', 'transferred'], inplace=True)

    # 合并 带有位置信息的训练集 与 地区假日信息 为 标准训练集
    valid_holiday.rename(columns={'locale_name': 'city', 'type': 'events'}, inplace=True)
    format_train = pd.merge(train_state, valid_holiday, how='left', left_on=['date', 'city'], right_on=['date', 'city'])
    valid_holiday.rename(columns={'city': 'state'}, inplace=True)
    format_train = pd.merge(format_train, valid_holiday, how='left', left_on=['date', 'state'], right_on=['date', 'state'])

    # 处理标准训练集假期数据
    event_nan = format_train[format_train['events_x'].isna()]
    event_not_nan = format_train[format_train['events_x'].notna()]
    # - 拼接假期数据
    event_nan['events_x'].fillna(value='', inplace=True)
    event_nan['events_y'].fillna(value='', inplace=True)
    event_nan['events_x'] = event_nan['events_x'] + event_nan['events_y']
    format_train = pd.concat([event_nan, event_not_nan], axis=0).sort_index()
    format_train.drop(axis=1, columns=['city', 'state', 'events_y'], inplace=True)
    format_train.rename(columns={'events_x': 'events'}, inplace=True)
    # format_train['events'] = format_train['events'].str.replace('', 'Nothing') # 替换空事件名称
    # for index, row in format_train.iterrows():
    #     if isnan(row['events_x']):
    #         if isnan(row['events_y']):
    #             format_train.iloc[index, -2] = 'Nothing'
    #         else:
    #             format_train.iloc[index, -2] = row['events_y']

    # 处理标准训练集无效数据
    format_train.fillna(0)
    # format_train.dropna(inplace=True)

    # 拼接油价数据



    format_train, format_test = format_data(train, transaction, extra_holidays, extra_oil, extra_store_inf, test)
    show_columns('format_train', format_train)
    show_columns('format_test', format_test)
