import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def difference(dataset, interval=1):
    """
    create difference list
    :param dataset:
    :param interval: start
    :return: diff list
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


def inverse_difference(last_ob, value):
    """
    :param last_ob: origin data
    :param value: diff data
    :return:
    """
    return value + last_ob


def fixna(data: pd.DataFrame):
    data.fillna(method='pad', inplace=True)
    data.fillna(method='bfill', inplace=True)


def data_mapping(data):
    object_cols = [cname for cname in data.columns
                   if data[cname].dtype == "object"
                   and cname != "date"]
    ordinal_encoder = OrdinalEncoder()
    data[object_cols] = ordinal_encoder.fit_transform(data[object_cols])
    return object_cols, data


def label_encoder(data: pd.DataFrame, columns: list):
    le = LabelEncoder()
    for col in columns:
        data[col] = le.fit_transform(data[col])


def format_data(dataset, transaction, holidays, oil, store):
    # 清理商店无用信息
    valid_extra_store_inf = store.drop(axis=1, columns=['type', 'cluster'])

    # 合并得到带有位置信息的训练集
    data_state = pd.merge(dataset, valid_extra_store_inf, how='left', left_on='store_nbr', right_on='store_nbr')

    # 清理无效假日，删除无用数据
    valid_holiday = holidays.drop(holidays[holidays['transferred']].index)
    valid_holiday.drop(axis=1, columns=['locale', 'description', 'transferred'], inplace=True)

    # 合并 带有位置信息的训练集 与 地区假日信息 为 标准训练集
    valid_holiday.rename(columns={'locale_name': 'city', 'type': 'events'}, inplace=True)
    format_datas = pd.merge(data_state, valid_holiday, how='left', left_on=['date', 'city'], right_on=['date', 'city'])
    valid_holiday.rename(columns={'city': 'state'}, inplace=True)
    format_datas = pd.merge(format_datas, valid_holiday, how='left', left_on=['date', 'state'],
                            right_on=['date', 'state'])

    # 处理标准训练集假期数据
    event_nan = format_datas[format_datas['events_x'].isna()]
    event_not_nan = format_datas[format_datas['events_x'].notna()]

    # - 拼接假期数据
    event_nan['events_x'].fillna(value='', inplace=True)
    event_nan['events_y'].fillna(value='', inplace=True)
    event_nan['events_x'] = event_nan['events_x'] + event_nan['events_y']
    format_datas = pd.concat([event_nan, event_not_nan], axis=0).sort_index()
    format_datas.drop(axis=1, columns=['city', 'state', 'events_y'], inplace=True)
    format_datas.rename(columns={'events_x': 'events'}, inplace=True)

    # 处理标准训练集无效数据
    fixna(format_datas)
    # format_datas.dropna(inplace=True)

    # 拼接油价数据
    fixna(oil)
    format_datas = pd.merge(format_datas, oil, how='left', left_on='date', right_on='date')

    # 数据编码
    columns = ['family', 'events']
    label_encoder(format_datas, columns)

    # 返回标准dataset
    return format_datas







    # format_datas['events'] = format_datas['events'].str.replace('', 'Nothing') # 替换空事件名称
    # for index, row in format_datas.iterrows():
    #     if isnan(row['events_x']):
    #         if isnan(row['events_y']):
    #             format_datas.iloc[index, -2] = 'Nothing'
    #         else:
    #             format_datas.iloc[index, -2] = row['events_y']
