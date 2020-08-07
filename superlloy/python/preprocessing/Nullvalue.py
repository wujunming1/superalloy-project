from pandas import DataFrame
import numpy as np
import time
import sys
import collections
from Nullcheck import read_data, Connectdatabase, null_check
from sklearn.impute import SimpleImputer

#读取数据文件后构建文件的类
class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

def null_drop(df, null_type, null_percent):
    """
    取出缺失值的方法
    :param df: Dataframe格式的原始数据集
    :param null_type: 空值的类型
    :param null_percent: 当空值占比达到该列的多少时需要先将该列删除
    :return:
    """
    new_df = None
    null_xy = null_check(df, null_type)
    null_dict = collections.defaultdict(int)
    for n in null_type:
        df = df.replace(n, np.nan)
    for x, y in null_xy:
        null_dict[df.columns.values[x]] += 1
    for col_name, null in null_dict.items():
        if null / df.shape[0] >= null_percent:
            new_df = df.drop(col_name, axis=1)
    return new_df.dropna(axis=0)

def null_completer(df, null_type, method,fill_value=None):
    """
    缺失值填充
    :param df: 原始Dataframe格式数据集
    :param null_type: 缺失值类型
    :param method: 填充方式
    :param fill_value: 填充值,只有method为constant时有效
    :return: 填充好的新的Dataframe格式数据
    """
    for n in null_type:
        df = df.replace(n, np.nan)
    if method != 'constant':
        imp = SimpleImputer(missing_values=np.nan, strategy=method)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=method,fill_value=fill_value)
    imp.fit(df.ix[:,:-1])#对于监督学习，最后一列不能填充
    mat = imp.transform(df.ix[:,:-1])
    new_df = DataFrame(mat, columns=df.columns.values[:-1])
    col = str(df.columns.values[-1])
    new_df[col] = df.pop(df.columns.values[-1])
    return new_df

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
        "data": data.values.tolist(),
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
    })
    print('End Insert')


if __name__ == '__main__':
    length_argv=len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    
    null_type = parameterlist[3]
    null_type = null_type.split(',')
    null_para = float(parameterlist[5])
    null_method = parameterlist[4]
    finish_data = None
    work_data = read_data(parameterlist[1])
    if null_method == 'drop':
        finish_data = null_drop(work_data, null_type, null_para)
    else:
        finish_data = null_completer(work_data, null_type, null_method, null_para)
    insert(parameterlist[0],parameterlist[1],parameterlist[2],parameterlist[6],finish_data)