import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import extension
import errors
import time
import pymongo

def read_data(path):
    """
    读取路径下的文件数据
    :param path: 文件路径
    :return: DataFrame类型的数据
    """
    try:
        file_extension = extension.file_extension(path)
        if file_extension == "csv":
            data_frame = pd.read_csv(path)
        elif file_extension == "xls" or file_extension == "xlsx":
            data_frame = pd.read_excel(path)
        elif file_extension == "txt":
            csv = np.loadtxt(path)
            data_frame = pd.DataFrame(csv)
        else:
            raise errors.DataTypeError
        return data_frame
    except Exception as e:
        print(e)

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

def encode_data(pre_data, one_hot_columns, binary_columns, mapper):
    """
    数据编码
    :param pre_data: DataFrame类型数据
    :param one_hot_columns: OneHot编码的列名列表
    :param binary_columns: 数值化编码的列名列表
    :param mapper: 数值化编码的对应关系
    :return: 清洗过后的数据
    """
    if one_hot_columns is not None:
        for column in one_hot_columns:
            pre_data = one_hot(pre_data, column)
    if binary_columns or mapper is not None:
        for column, map in zip(binary_columns, mapper):
            pre_data = binary(pre_data, column, map)
    return pre_data

def Connectdatabase():
    """
    连接MongoDB数据库
    :return: 无
    """
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def insert(file_name,file_path,alg,username,finish_data):
    """
        插入预处理之后的数据到mongodb
        :param file_name: 文件名
        :param file_path: 文件的绝对路径
        :param alg: 预处理方法
        :param username: 进行预处理的用户名
        :param finish_data：已经完成预处理之后的数据
        :return: 无
    """
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data = finish_data
    feature_names = finish_data.columns.values
    feature_count = finish_data.shape[1]

    print('Insert DataBase:')
    db = Connectdatabase()
    db.PreprocessedData.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "data": data.astype(np.float64).values.tolist(),
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
    })
    print('End Insert')

if __name__ == "__main__":
    try:
        length_argv = len(sys.argv)
        print(length_argv)
        parameterlist = []
        for i in range(1, len(sys.argv)):
            para = sys.argv[i]
            parameterlist.append(para)
        print(parameterlist)
        onehot = sys.argv[4]
        numerical = sys.argv[5]
        numerical_dict = sys.argv[6]
        numerical_dict_list = []#用于存放所有数值化编码的列的映射关系，列表中的每一个元素都是字典
        if onehot == "none":
            onehot = None
        else:
            onehot = onehot.split(",")
        if numerical == "none":
            numerical = None
        else:
            numerical = numerical.split(",")
        if numerical_dict == "none":
            numerical_dict_list = None
        else:
            numerical_dict = numerical_dict.split(";")
            for d in numerical_dict:
                d = d.split(",")
                print(d)
                dict1 = {}
                for i in d:
                    t = i.split(":")
                    print(t)
                    dict1[t[0]] = t[1]
                numerical_dict_list.append(dict1)
        
        work_data = read_data(parameterlist[1])
        work_data = clean_space(work_data)
        finish_data = encode_data(work_data, onehot, numerical, numerical_dict_list)
        insert(parameterlist[0],parameterlist[1],parameterlist[2],parameterlist[6],finish_data)
    except Exception as e:
        print(e)