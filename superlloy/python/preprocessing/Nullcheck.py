import pandas as pd
import numpy as np
import extension
import errors
import pymongo
import time
import sys

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

def null_check(df, null_type):
    """
        获取数据集中的所有类型空值的位置
        :param df: Dataframe数据
        :param null_type: 被判定为空值的类型
        :return:空值坐标
    """
    for n in null_type:
        df = df.replace(n, np.nan)
    m = np.matrix(df.isna())
    result = np.argwhere(m == True)
    #交换空值所在位置的行列坐标
    r1 = result[:,1].copy()
    r2 = result[:,0].copy()
    result[:,1] = r2
    result[:,0] = r1
    return result

def Connectdatabase():
    """
    连接MongoDB数据库
    :return: 无
    """
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def insert(file_name,alg,username,xy,finish_data):
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
    feature_names = finish_data.columns.values
    feature_count = finish_data.shape[1]

    print('Insert DataBase:')
    db = Connectdatabase()
    db.PreprocessedData.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
        "xy": xy.astype(np.int).tolist()
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
    work_data = read_data(parameterlist[1])
    xy = null_check(work_data, null_type)
    insert(parameterlist[0], parameterlist[2], parameterlist[4], xy, work_data)