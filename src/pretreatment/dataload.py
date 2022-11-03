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

    print(get_data_list(extra_holidays, 2))
    print(get_data_list(extra_holidays, 3))

    # combine train data and test data
    train_len = train.shape[0]
    test.insert(loc=4, column='sales', value=0)
    combine = train.append(test, ignore_index=True)
    # train_data = train.groupby(['date']).agg({'sales': 'mean', 'onpromotion': 'mean'})

    # event data clearning
    valid_holiday = extra_holidays.drop(extra_holidays[extra_holidays['transferred']].index)
    # combine store and holiday
    valid_holiday.drop(axis=1, columns=['locale', 'description', 'transferred'])
    #valid_holiday.insert(loc=valid_holiday.shape[1], column='')



    format_train, format_test = format_data(train, transaction, extra_holidays, extra_oil, extra_store_inf, test)
    show_columns('format_train', format_train)
    show_columns('format_test', format_test)
