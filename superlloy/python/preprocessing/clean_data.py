import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def clean_space(data):
    data.dropna(inplace=True)
    return data

def binary(data, name, mapper):
    """
    对DataFrame数据进行数值化编码
    :param data: DataFrame类型数据
    :param name: string类型数据,代表列名
    :param mapper: dict类型数据，代表数值替换
    :return: 编码后数据
    """
    data[name] = data[name].map(mapper)
    return data

def one_hot(data, name):
    """
    对DataFrame数据进行OneHot编码
    :param data:DataFrame类型数据
    :param name:string类型数据,代表列名
    :return: 编码后数据
    """
    columns = []
    set_example = set(data[name].values)
    num = len(set_example)
    for i in range(num):
        columns.append(name + '_' + str(i))
    encoder = LabelEncoder()
    sex = encoder.fit_transform(data[name].values)
    sex = np.array([sex]).T
    encoder = OneHotEncoder()
    result = encoder.fit_transform(sex)
    result = result.toarray()
    data = pd.concat([pd.DataFrame(result, columns=columns), data], axis=1)
    data = data.drop([name], axis=1)
    return data

