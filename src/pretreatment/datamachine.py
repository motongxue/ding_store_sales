import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


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


def data_mapping(data):
    object_cols = [cname for cname in data.columns
                   if data[cname].dtype == "object"
                   and cname != "date"]
    ordinal_encoder = OrdinalEncoder()
    data[object_cols] = ordinal_encoder.fit_transform(data[object_cols])
    return object_cols, data


def reshape_holiday(holiday):
    hld = np.array(holiday)
    new_hld = np.empty([0, 8], order='C')

    return pd.DataFrame(new_hld, columns=['date',
                                          'city',
                                          'Additional', 'Bridge', 'Event', 'Holiday', 'Transfer', 'Work Day'])


def format_data(train, transaction, holidays, oil, store, test):
    holidays.rename(columns={'date': 'date',
                             'type': 'Daily_holiday_type',
                             'locale': 'Daily_holiday_locale',
                             'locale_name': 'Daily_holiday_locale_name',
                             'description': "Daily_holiday_description",
                             'transferred': "Daily_holiday_transferred"},
                    inplace=True)
    store.rename(columns={'store_nbr': 'store_nbr',
                          'city': 'stores_city',
                          'state': 'store_state',
                          'type': 'store_type',
                          'cluster': 'store_cluster'},
                 inplace=True)
    transaction.rename(columns={'transactions': 'Daily_transactions'})
    new_train = pd.merge(train, holidays, how='left', left_on='date', right_on='date')
    new_test = pd.merge(test, holidays, how='left', left_on='date', right_on='date')
    new_train = pd.merge(new_train, oil, how='left', left_on='date', right_on='date')
    new_test = pd.merge(new_test, oil, how='left', left_on='date', right_on='date')
    new_train = pd.merge(new_train, store, how='left', left_on='store_nbr', right_on='store_nbr')
    new_test = pd.merge(new_test, store, how='left', left_on='store_nbr', right_on='store_nbr')
    new_train = pd.merge(new_train, transaction, how='left', on=['date', 'store_nbr'])
    new_test = pd.merge(new_test, transaction, how='left', on=['date', 'store_nbr'])
    return new_train, new_test
